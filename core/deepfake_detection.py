import os
import cv2
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from timm import create_model
from torchvision import transforms
from textblob import TextBlob
from torch_geometric.data import HeteroData
from hetero_gnn_model import HeteroGNN
from user_interaction_modeler import UserInteractionModeler
from sklearn.preprocessing import StandardScaler
import yt_dlp


def compute_media_score(video_path, model_path, frame_interval=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model('xception', pretrained=False, num_classes=2, drop_rate=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx, records = 0, []
    with torch.no_grad():
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc='Frames')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(img)
                tensor = preprocess(pil).unsqueeze(0).to(device)
                prob = torch.softmax(model(tensor), dim=1)[0]
                pred = prob.argmax().item()
                records.append({'frame_index': frame_idx, 'fake_prob': float(prob[1]), 'prediction': int(pred)})
            frame_idx += 1
            pbar.update(1)
        pbar.close()
    cap.release()
    df = pd.DataFrame(records)
    return df['prediction'].mean(), df


def fetch_youtube_metadata(video_url):
    ydl_opts = {'quiet': True, 'skip_download': True, 'extract_flat': False, 'forcejson': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        return {
            'video_id': info.get('id'),
            'title': info.get('title'),
            'description': info.get('description', ''),
            'upload_date': info.get('upload_date'),
            'duration_seconds': info.get('duration'),
            'view_count': info.get('view_count', 0),
            'like_count': info.get('like_count', 0),
            'comment_count': info.get('comment_count', 0),
            'uploader': info.get('uploader'),
            'category': info.get('categories')[0] if info.get('categories') else 'Unknown',
            'engagement_rate': (info.get('like_count', 0) + info.get('comment_count', 0)) / max(info.get('view_count', 1), 1)
        }


def build_hetero_data(interactions_df, user_feat_df, video_feat_df):
    interactions_df['user_id'] = interactions_df['user_id'].astype(str)
    interactions_df['video_id'] = interactions_df['video_id'].astype(str)
    user_feat_df.index = user_feat_df.index.astype(str)
    video_feat_df.index = video_feat_df.index.astype(str)

    user_ids = user_feat_df.index.unique()
    video_ids = video_feat_df.index.unique()
    user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
    video2idx = {vid: idx for idx, vid in enumerate(video_ids)}

    def build_node_tensor(df, use_cols):
        feats = df[use_cols]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feats.values)
        return torch.tensor(scaled, dtype=torch.float)

    user_feat_cols = [
        'age_group_enc', 'activity_level_enc', 'like_prob', 'comment_prob',
        'share_prob', 'session_length_norm', 'videos_per_session_norm',
        'preferred_categories_count'  # ✅ total: 8
    ]

    expected_video_columns = [
    'log_views', 'log_likes', 'log_comments', 'duration_hours',
    'engagement_rate', 'like_ratio', 'desc_length_k',
    'tags_count', 'is_live', 'age_limit_norm'
    ]

    # Ensure all required columns are present; fill missing ones if needed
    for col in expected_video_columns:
        if col not in video_feat_df.columns:
            video_feat_df[col] = 0.0

    video_feat_df = video_feat_df[expected_video_columns]

    user_x = build_node_tensor(user_feat_df, user_feat_cols)
    video_x = build_node_tensor(video_feat_df, expected_video_columns)

    u_idx = interactions_df['user_id'].map(user2idx)
    v_idx = interactions_df['video_id'].map(video2idx)
    valid = u_idx.notna() & v_idx.notna()
    u_idx = u_idx[valid].astype(int).to_numpy()
    v_idx = v_idx[valid].astype(int).to_numpy()

    edge_index = torch.tensor([u_idx, v_idx], dtype=torch.long)
    edge_attr_cols = ['watch_time_ratio', 'engagement_strength']
    edge_attr = torch.tensor(interactions_df.loc[valid, edge_attr_cols].values, dtype=torch.float)

    rev_edge_index = edge_index[[1, 0], :]
    rev_edge_attr = edge_attr.clone()

    data = HeteroData()
    data['user'].x = user_x
    data['video'].x = video_x
    data['user', 'interacts', 'video'].edge_index = edge_index
    data['user', 'interacts', 'video'].edge_attr = edge_attr
    data['video', 'rev_interacts', 'user'].edge_index = rev_edge_index
    data['video', 'rev_interacts', 'user'].edge_attr = rev_edge_attr

    return data



def compute_graph_score(metadata_dict, gnn_model_path):
    videos_df = pd.DataFrame([metadata_dict])
    modeler = UserInteractionModeler()
    users_df = modeler.generate_user_base(videos_df)
    interactions_df = modeler.generate_interactions(videos_df, users_df)

    # ✅ Use GNNDataPreparator to generate engineered features
    from gnn_data_preparator import GNNDataPreparator  # adjust if it's in the same file

    preparator = GNNDataPreparator()
    preparator.create_graph(videos_df, users_df, interactions_df)
    video_feat_df, user_feat_df = preparator.prepare_node_features()
    print("🧾 Sample user_feat_df.index:", user_feat_df.index[:5])
    print("🧾 Sample video_feat_df.index:", video_feat_df.index[:5])
    print("🧾 Sample interactions_df['user_id']:", interactions_df['user_id'].unique()[:5])
    print("🧾 Sample interactions_df['video_id']:", interactions_df['video_id'].unique()[:5])

    # Use actual IDs from interactions for indexing
    user_feat_df = user_feat_df.set_index('user_id')
    video_feat_df = video_feat_df.set_index('video_id')

    # (Optional) Slice top 8 video features used during training
    video_feat_df = video_feat_df[[
    'log_views', 'log_likes', 'log_comments', 'duration_hours',
    'engagement_rate', 'like_ratio', 'desc_length_k',
    'tags_count', 'is_live', 'age_limit_norm']]

    data = build_hetero_data(interactions_df, user_feat_df, video_feat_df)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeteroGNN().to(device)
    model.load_state_dict(torch.load(gnn_model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
        out_dict = model(x_dict, edge_index_dict)

        edge_index = data['user', 'interacts', 'video'].edge_index
        user_embed = out_dict['user'][edge_index[0]]
        video_embed = out_dict['video'][edge_index[1]]
        preds = (user_embed * video_embed).sum(dim=1)
        print("▶ user_embed shape:", user_embed.shape)
        print("▶ video_embed shape:", video_embed.shape)
        print("▶ Any NaNs in user_embed?", torch.isnan(user_embed).any().item())
        print("▶ Any NaNs in video_embed?", torch.isnan(video_embed).any().item())
        print("▶ preds shape:", preds.shape)
        print("▶ Any NaNs in preds?", torch.isnan(preds).any().item())
        print("▶ preds mean:", preds.mean().item() if preds.numel() > 0 else "EMPTY")

        graph_score = preds.mean().item()

    return graph_score, interactions_df



def compute_sentiment_score(metadata_dict, sentiment_model_path, feature_list_path):
    # Convert metadata_dict to DataFrame
    features = pd.DataFrame([metadata_dict])
    
    # Preprocess features to match training
    # --- Add derived features like in training ---
    features['description_length'] = features['description'].str.len()
    features['comment_rate'] = features['comment_count'] / (features['view_count'] + 1e-5)

    # Drop irrelevant columns
    drop_cols = ['video_id', 'title', 'upload_date', 'uploader', 'language', 'category', 'description', 'engagement_rate']
    features = features.drop(columns=[col for col in drop_cols if col in features.columns], errors='ignore')

    # Load model and expected feature names
    model = joblib.load(sentiment_model_path)
    expected_features = joblib.load(feature_list_path)

    # Fill missing expected features with 0
    for col in expected_features:
        if col not in features.columns:
            features[col] = 0

    # Ensure correct column order
    features = features[expected_features]

    # Predict and return sentiment score
    sentiment_log = model.predict(features)[0]
    return np.expm1(sentiment_log)  # Reverse log1p transformation


def check_bot_flag(users_df, bot_model_path, bot_feature_list_path="bot_model_features.pkl"):
    model = joblib.load(bot_model_path)
    expected_features = joblib.load(bot_feature_list_path)

    # Assume your users_df is preprocessed or has raw features — add encodings here if needed:
    if 'activity_level_enc' not in users_df.columns:
        users_df['activity_level_enc'] = users_df['activity_level'].map({'low': 0, 'medium': 1, 'high': 2}).fillna(0)

    if 'age_group_enc' not in users_df.columns:
        users_df['age_group_enc'] = users_df['age_group'].map({'child': 0, 'teen': 1, 'adult': 2, 'senior': 3}).fillna(0)

    # Rename or create features to match training:
    rename_map = {
        'comment_probability': 'comment_prob',
        'like_probability': 'like_prob'
    }
    users_df = users_df.rename(columns=rename_map)

    # Drop unused or extra columns
    X = users_df.drop(columns=['user_id'], errors='ignore')

    # Fill in any missing expected columns
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0.0

    # Ensure correct order
    X = X[expected_features]

    # Predict probabilities
    probas = model.predict_proba(X)[:, 1]
    bot_prob = float(np.mean(probas))
    return bot_prob > 0.5


# ================================
# VIDEO DOWNLOAD
# ================================
def download_video(youtube_url, output_path="downloaded_video.mp4"):
    ydl_opts = {
        'quiet': True,
        'outtmpl': output_path,
        'format': 'mp4/best', 
        #'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"⬇️  Downloading video from {youtube_url}...")
        ydl.download([youtube_url])
    print(f"✅ Video downloaded to {output_path}")
    return output_path



def evaluate_video(youtube_url, model_paths):
    print("\n🚀 Starting video evaluation pipeline...\n")
    video_path = download_video(youtube_url)
    media_score, media_df = compute_media_score(video_path, model_paths['cnn'])
    metadata = fetch_youtube_metadata(youtube_url)
    metadata_df = pd.DataFrame([metadata])

    graph_score, interactions_df = compute_graph_score(metadata, model_paths['gnn'])
    sentiment_score = compute_sentiment_score(metadata, model_paths['sentiment'], model_paths['feature_list_path'])

    # Extract synthetic users again for bot model
    modeler = UserInteractionModeler()
    users_df = modeler.generate_user_base(pd.DataFrame([metadata]))
    bot_flag = check_bot_flag(users_df, model_paths['bot'], model_paths['bot_feature_list_path'])

    results = {
        "media_score": media_score,
        "graph_score": graph_score,
        "sentiment_score": sentiment_score,
        "bot_flag": bot_flag,
        "metadata": metadata
    }

    print("\n🎉 Final Evaluation Result:")
    for k, v in results.items():
        if k != "metadata":
            print(f" - {k}: {v}")
    return results, media_df, interactions_df, metadata_df

if __name__ == "__main__":
    model_paths = {
        "cnn": "D:/Deepfake_Detection/models/best_model.pth",
        "gnn": "D:/Deepfake_Detection/models/best_heterognn_model1.pt",
        "sentiment": "D:/Deepfake_Detection/models/sentiment_model_gb1.pkl",
        "bot": "D:/Deepfake_Detection/models/bot_model_tuned1.pkl",
        "feature_list_path":"D:/Deepfake_Detection/models/sentiment_model_features.pkl",
        "bot_feature_list_path":"D:/Deepfake_Detection/models/bot_model_features.pkl"
    }

    evaluate_video("https://www.youtube.com/shorts/7-4QEinx930", model_paths)