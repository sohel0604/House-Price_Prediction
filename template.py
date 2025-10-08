import os

# List of all project file paths to be created
list_of_files = [
    
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",
    "src/logger.py",
    "src/exception.py",
    "src/utils.py"
    
]

for filepath in list_of_files:
    # Normalize path (to work on Windows/Linux)
    filepath = os.path.normpath(filepath)
    
    # Split directory and filename
    filedir, filename = os.path.split(filepath)

    # Create directories if they don’t exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        print(f"Created directory: {filedir}")

    # Create empty file if not exists
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        print(f"Created empty file: {filepath}")
    else:
        print(f"File already exists: {filepath}")
