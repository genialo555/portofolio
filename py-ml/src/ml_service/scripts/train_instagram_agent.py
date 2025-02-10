async def train_instagram_agent():
    """Fine-tune l'agent pour Instagram."""
    tuner = ModelFineTuner(dataset_path="data/instagram_interactions")
    
    # Configuration spécifique Instagram
    instagram_config = {
        'model_key': 'instagram_specialist',
        'subset_filters': {
            'domain': 'social_media',
            'platform': 'instagram',
            'task_type': ['content_strategy', 'monetization', 'growth']
        },
        'training_args': {
            'num_train_epochs': 3,
            'learning_rate': 1e-4,
            'per_device_train_batch_size': 4
        }
    }
    
    await tuner.train(**instagram_config) 