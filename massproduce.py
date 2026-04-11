import asyncio
from pipeline import ResearchPipeline
from dotenv import load_dotenv
load_dotenv()

async def main():
    """
    topics = [
        "The impact of climate change on global agriculture",
        "The role of artificial intelligence in healthcare",]
    """
    topics = [
        "The impact of microplastic pollution on ecological environment including physical, chemical, and biological effects.",
        "The effectiveness of remote work policies on employee productivity and well-being",
        "How large language models are transforming software development practices",
        "The role of social media in shaping public opinion during elections",
        "The impact of renewable energy adoption on global carbon emissions",
        "Mental health impacts of social media use among teenagers",
        "The connection between gut microbiome and mental health disorders",
        "The influence of video games on cognitive development in children",
        "The role of blockchain technology in enhancing supply chain transparency",
        "The impact of 5G technology on internet connectivity and economic growth"
    ]

    for i, topic in enumerate(topics):
        pipeline = ResearchPipeline(topic, 8, filepath=f"contexts_outputs/contexts_{i+2}.json")
        async for event in pipeline.run():
            print(f"[{i}] {event.get('type', 'unknown')}")

if __name__ == "__main__":
    asyncio.run(main())