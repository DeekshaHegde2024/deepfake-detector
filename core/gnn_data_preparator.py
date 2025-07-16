import pandas as pd
import numpy as np
import networkx as nx



class GNNDataPreparator:
    """Prepare data for Graph Neural Network"""
    def __init__(self):
        self.graph = nx.Graph()
        self.node_features = {}
        self.edge_features = {}

    def create_graph(self, videos_df, users_df, interactions_df):
        """Create bipartite graph with videos and users as nodes"""
        print("🕸️ Creating graph structure...")
        # Add video nodes
        for _, video in videos_df.iterrows():
            self.graph.add_node(
                video['video_id'],
                node_type='video',
                **video.to_dict()
            )
        # Add user nodes
        for _, user in users_df.iterrows():
            self.graph.add_node(
                user['user_id'],
                node_type='user',
                **user.to_dict()
            )
        # Add edges from interactions
        for _, interaction in interactions_df.iterrows():
            user_id = interaction['user_id']
            video_id = interaction['video_id']
            if self.graph.has_edge(user_id, video_id):
                # Update existing edge (user interacted with video multiple times)
                edge_data = self.graph[user_id][video_id]
                edge_data['interaction_count'] += 1
                edge_data['total_watch_time'] += interaction['watch_time_seconds']
                edge_data['avg_engagement'] = (edge_data['avg_engagement'] + interaction['engagement_strength']) / 2
            else:
                # Add new edge
                self.graph.add_edge(
                    user_id,
                    video_id,
                    interaction_count=1,
                    total_watch_time=interaction['watch_time_seconds'],
                    avg_engagement=interaction['engagement_strength'],
                    first_interaction=interaction['timestamp'],
                    **interaction.to_dict()
                )
        print(f"✅ Created graph with {self.graph.number_of_nodes():,} nodes and {self.graph.number_of_edges():,} edges")
        return self.graph

    def prepare_node_features(self):
        """Prepare feature matrices for nodes"""
        print("🎯 Preparing node features...")
        video_features = []
        user_features = []
        video_nodes = []
        user_nodes = []
        for node, data in self.graph.nodes(data=True):
            if data['node_type'] == 'video':
                video_nodes.append(node)
                features = [
                    np.log1p(data.get('view_count', 0)),
                    np.log1p(data.get('like_count', 0)),
                    np.log1p(data.get('comment_count', 0)),
                    data.get('duration_seconds', 0) / 3600, # Hours
                    data.get('engagement_rate', 0),
                    data.get('like_ratio', 0),
                    data.get('description_length', 0) / 1000, # Normalize
                    data.get('tags_count', 0),
                    1 if data.get('is_live', False) else 0,
                    data.get('age_limit', 0) / 18, # Normalize
                    # Add sentiment score (target for sentiment model)
                    data.get('sentiment_score', 0) # This will be added by DataAnalyzer
                ]
                video_features.append(features)
            elif data['node_type'] == 'user':
                user_nodes.append(node)
                features = [
                    {'teen': 0, 'young_adult': 1, 'adult': 2, 'senior': 3}.get(data.get('age_group', 'adult'), 2),
                    {'low': 0, 'medium': 1, 'high': 2}.get(data.get('activity_level', 'medium'), 1),
                    data.get('like_probability', 0),
                    data.get('comment_probability', 0),
                    data.get('share_probability', 0),
                    data.get('session_length', 0) / 100, # Normalize
                    data.get('videos_per_session', 0) / 20, # Normalize
                    len(data.get('preferred_categories', [])),
                    # Add bot flag (target for bot detection model)
                    1 if data.get('is_bot', False) else 0 # This will be added by UserInteractionModeler
                ]
                user_features.append(features)

        video_feature_columns = ['log_views', 'log_likes', 'log_comments', 'duration_hours',
                                 'engagement_rate', 'like_ratio', 'desc_length_k', 'tags_count',
                                 'is_live', 'age_limit_norm', 'sentiment_score']
        video_features_df = pd.DataFrame(video_features, index=video_nodes, columns=video_feature_columns)

        user_feature_columns = ['age_group_enc', 'activity_level_enc', 'like_prob',
                                'comment_prob', 'share_prob', 'session_length_norm',
                                'videos_per_session_norm', 'preferred_categories_count', 'is_bot']
        user_features_df = pd.DataFrame(user_features, index=user_nodes, columns=user_feature_columns)

        video_features_df.index.name = 'video_id'
        video_features_df.reset_index(inplace=True)

        user_features_df.index.name = 'user_id'
        user_features_df.reset_index(inplace=True)

        return video_features_df, user_features_df


    def prepare_edge_features(self):
        """Prepare edge feature matrix"""
        print("🔗 Preparing edge features...")
        edge_data = []
        edge_index = []
        for u, v, data in self.graph.edges(data=True):
            edge_index.append([u, v])
            features = [
                data.get('interaction_count', 1),
                data.get('total_watch_time', 0) / 3600, # Hours
                data.get('avg_engagement', 0),
                data.get('watch_time_ratio', 0),
                1 if 'like' in data.get('interaction_types', '') else 0,
                1 if 'comment' in data.get('interaction_types', '') else 0,
                1 if 'share' in data.get('interaction_types', '') else 0,
                # Device type as a categorical feature (one-hot or label encode later)
                data.get('device_type', 'unknown')
            ]
            edge_data.append(features)

        edge_features_df = pd.DataFrame(
            edge_data,
            columns=['interaction_count', 'total_watch_hours', 'avg_engagement',
                     'watch_time_ratio', 'has_like', 'has_comment', 'has_share', 'device_type']
        )
        edge_index_df = pd.DataFrame(edge_index, columns=['source', 'target'])
        return edge_features_df, edge_index_df