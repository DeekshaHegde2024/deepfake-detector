import numpy as np
import pandas as pd
import random
from tqdm import tqdm


class UserInteractionModeler:
    """Generate synthetic user interactions based on video metadata"""
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.user_profiles = {}
        self.interaction_patterns = {}

    def generate_user_base(self, videos_df, num_users=None):
        """Generate synthetic user base with realistic behavior patterns"""
        if num_users is None:
            # Scale users based on total engagement
            total_engagement = videos_df['like_count'].sum() + videos_df['comment_count'].sum()
            num_users = min(int(total_engagement / 100), 50000) # Cap at 50k users
            num_users = max(num_users, 1000) # Minimum 1k users
        print(f"🧑‍🤝‍🧑 Generating {num_users:,} synthetic users...")
        users = []
        for i in tqdm(range(num_users), desc="Creating users"):
            user_id = f"user_{i:06d}"
            # User demographics and preferences
            age_group = np.random.choice(['teen', 'young_adult', 'adult', 'senior'],
                                         p=[0.15, 0.35, 0.35, 0.15])
            activity_level = np.random.choice(['low', 'medium', 'high'],
                                              p=[0.4, 0.4, 0.2])
            # Content preferences based on categories
            categories = videos_df['category'].unique()
            preferred_categories = np.random.choice(categories,
                                                    size=min(3, len(categories)),
                                                    replace=False).tolist()
            # Behavioral patterns
            like_probability = np.random.beta(2, 5) # Most users like sparingly
            comment_probability = np.random.beta(1, 10) # Comments are rarer
            share_probability = np.random.beta(1, 20) # Shares are very rare
            # Viewing patterns
            session_length = np.random.lognormal(2, 1) # Average session time
            videos_per_session = max(1, int(np.random.poisson(5)))

            # SIMULATE BOT FLAG: Introduce a small percentage of bots
            is_bot = random.random() < 0.05 # 5% of users are bots
            if is_bot:
                # Bots might have slightly different characteristics for flagging
                activity_level = 'high' # Bots often have high activity
                like_probability = np.random.uniform(0.01, 0.1) # Bots might like less or more consistently
                comment_probability = np.random.uniform(0.001, 0.05) # Bots comment rarely or with generic text
                share_probability = np.random.uniform(0.001, 0.02)
                session_length = np.random.uniform(10, 60) # Shorter, more consistent sessions
                videos_per_session = max(1, int(np.random.poisson(10))) # More videos per session
                device_type_bias = np.random.choice(['desktop', 'server'], p=[0.7, 0.3]) # Bots might use specific devices/environments

            user = {
                'user_id': user_id,
                'age_group': age_group,
                'activity_level': activity_level,
                'preferred_categories': preferred_categories,
                'like_probability': like_probability,
                'comment_probability': comment_probability,
                'share_probability': share_probability,
                'session_length': session_length,
                'videos_per_session': videos_per_session,
                'join_date': self._random_date('2020-01-01', '2023-12-31'),
                'is_bot': is_bot # Add bot flag
            }
            if is_bot:
                 user['device_type_bias'] = device_type_bias # Specific bot device type
            users.append(user)
            self.user_profiles[user_id] = user
        return pd.DataFrame(users)

    def generate_interactions(self, videos_df, users_df, interaction_ratio=0.1):
        """Generate realistic user-video interactions"""
        print(f"🔗 Generating user-video interactions...")
        interactions = []
        # Calculate interaction probabilities for each video
        videos_df['interaction_weight'] = (
            np.log1p(videos_df['view_count']) * 0.4 +
            videos_df['engagement_rate'] * 0.3 +
            (100 - videos_df.get('duration_seconds', 300) / 10) * 0.1 + # Shorter videos get more interactions
            np.random.random(len(videos_df)) * 0.2 # Add some randomness
        )
        # Normalize weights
        videos_df['interaction_weight'] = videos_df['interaction_weight'] / videos_df['interaction_weight'].sum()

        # Generate interactions for each user
        for _, user in tqdm(users_df.iterrows(), total=len(users_df), desc="Creating interactions"):
            user_id = user['user_id']
            # Determine how many videos this user will interact with
            if user['activity_level'] == 'low':
                num_interactions = max(1, int(np.random.poisson(3)))
            elif user['activity_level'] == 'medium':
                num_interactions = max(1, int(np.random.poisson(10)))
            else: # high activity
                num_interactions = max(1, int(np.random.poisson(20))) # Reduced from 30 for faster generation

            if user['is_bot']: # Bots interact more consistently, less randomly
                 num_interactions = max(5, int(np.random.poisson(user['videos_per_session']))) # Use bot's specific rate

            # Bias selection towards user's preferred categories
            user_video_weights = videos_df['interaction_weight'].copy()
            for cat in user['preferred_categories']:
                cat_mask = videos_df['category'] == cat
                user_video_weights[cat_mask] *= 3 # 3x more likely to interact with preferred content
            # Renormalize
            user_video_weights = user_video_weights / user_video_weights.sum()

            # Sample videos for interaction
            selected_videos = np.random.choice(
                videos_df.index,
                size=min(num_interactions, len(videos_df)),
                replace=False,
                p=user_video_weights
            )
            # Generate interaction details for each selected video
            for video_idx in selected_videos:
                video = videos_df.loc[video_idx]
                # Determine interaction type
                interaction_types = []
                # View (always happens)
                interaction_types.append('view')
                # Like
                if np.random.random() < user['like_probability']:
                    interaction_types.append('like')
                # Comment
                if np.random.random() < user['comment_probability']:
                    interaction_types.append('comment')
                # Share
                if np.random.random() < user['share_probability']:
                    interaction_types.append('share')

                # Watch time (percentage of video watched)
                if video['duration_seconds'] > 0:
                    watch_time_ratio = min(1.0, np.random.beta(2, 3)) # Most people don't watch full videos
                    if user['is_bot']: # Bots might watch more consistently or very little
                        watch_time_ratio = np.random.uniform(0.1, 0.9) if np.random.random() < 0.8 else np.random.uniform(0.01, 0.05) # Either almost full or very little
                    watch_time = video['duration_seconds'] * watch_time_ratio
                else:
                    watch_time_ratio = np.random.beta(2, 3)
                    if user['is_bot']:
                         watch_time_ratio = np.random.uniform(0.1, 0.9) if np.random.random() < 0.8 else np.random.uniform(0.01, 0.05)
                    watch_time = 180 * watch_time_ratio # Assume 3min default

                # Engagement strength (composite score)
                engagement_strength = (
                    len(interaction_types) * 0.3 +
                    watch_time_ratio * 0.4 +
                    (1 if video['category'] in user['preferred_categories'] else 0) * 0.3
                )
                if user['is_bot']:
                    engagement_strength = np.random.uniform(0.1, 0.5) # Bots often have lower/less varied engagement strength
                    if 'comment' in interaction_types: # Bots comments might not signal strong engagement
                        engagement_strength *= 0.5


                interaction = {
                    'user_id': user_id,
                    'video_id': video['video_id'],
                    'interaction_types': ','.join(interaction_types),
                    'timestamp': self._random_timestamp(),
                    'watch_time_seconds': round(watch_time, 1),
                    'watch_time_ratio': round(watch_time_ratio, 3),
                    'engagement_strength': round(engagement_strength, 3),
                    'device_type': user.get('device_type_bias', np.random.choice(['mobile', 'desktop', 'tablet'], p=[0.6, 0.3, 0.1])), # Bots might have specific device_type_bias
                    'session_id': f"session_{np.random.randint(1000000, 9999999)}"
                }
                interactions.append(interaction)
        interactions_df = pd.DataFrame(interactions)
        print(f"✅ Generated {len(interactions_df):,} user interactions")
        return interactions_df

    def _random_date(self, start_date, end_date):
        """Generate random date between start and end"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        random_date = start + (end - start) * np.random.random()
        return random_date.strftime('%Y-%m-%d')

    def _random_timestamp(self):
        """Generate random timestamp in the last 2 years"""
        start = pd.Timestamp.now() - pd.Timedelta(days=730)
        end = pd.Timestamp.now()
        random_ts = start + (end - start) * np.random.random()
        return random_ts.strftime('%Y-%m-%d %H:%M:%S')