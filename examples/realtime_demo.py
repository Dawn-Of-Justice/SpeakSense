import asyncio
from speaksense.pipeline import RealtimePipeline
from speaksense.utils import load_config

async def main():
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Initialize pipeline
    pipeline = RealtimePipeline(config)
    
    # Process stream
    async for prediction in pipeline.process_stream():
        if prediction > config.pipeline.threshold:
            print("Speaking to VA detected!")
        else:
            print("No VA interaction detected.")

if __name__ == "__main__":
    asyncio.run(main())