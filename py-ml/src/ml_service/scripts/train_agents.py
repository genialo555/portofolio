async def main():
    tuner = ModelFineTuner(dataset_path="/path/to/your/100gb/dataset")
    
    # Fine-tune l'agent enseignant
    await tuner.train(
        'teacher',
        subset_filter={'type': 'teaching_interaction'}
    )
    
    # Fine-tune l'agent de débat
    await tuner.train(
        'debate',
        subset_filter={'type': 'debate_interaction'}
    ) 