import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import requests

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


def get_website_html(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching the URL {url}: {e}")
        return ""


def extract_core_website_content(html: str) -> str:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are an expert web content extractor (web scraper)."},
            {"role": "user", "content": f"""
                Your task is to extract the core content from a given HTML page.
                The core content should be the main text, excluding navigation, footers, and other 
                non-essential elements like scripts, etc. Here is the HTML content:
                <html>
                {html}
                </html>
                Please extract the core content and return it as plain text.
            """},
        ],
        temperature=1.0,
        stream=False
    )

    return response.choices[0].message.content


def summarize_content(content: str) -> str:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": f"""
                Your task is to summarize the provided content into a concise and clear summary in english.
                Here is the content to summarize:
                <content>
                {content}
                </content>
                Please provide a brief summary of the main points in the content. Prefer bullet points and 
                avoid unnecessary explanations.
            """},
        ],
        temperature=1.5,
        stream=False
    )

    return response.choices[0].message.content


def generate_x_post(summary: str) -> str:
    # Load examples from the JSON file
    with open("post-examples.json", "r", encoding="utf-8") as f:
        examples = json.load(f)

    examples_str = ""
    for i, example in enumerate(examples, 1):
        examples_str += f"""
        <example-{i}>
            <topic>
            {example['topic']}
            </topic>

            <generated-post>
            {example['post']}
            </generated-post>
        </example-{i}>
        """

    prompt = f"""
        Your task is to generate a post based on a short text summary in english.
        Your post must be concise and impactful.
        Avoid using hashtags and lots of emojis (a few emojis are okay, but not too many).
        Keep the post short and focused, structure it in a clean, readable way, using line breaks and empty lines 
        to enhance readability. Here's the text summary which you should use to generate the post:
        <summary>
        {summary}
        </summary>
        Here are some examples of topics and generated posts:
        <examples>
            {examples_str}
        </examples>
        Please use the tone, language, structure , and style of the examples provided above to generate a post that is 
        engaging and relevant to the topic provided by the user. Don't use the content from the examples!
    """

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are an expert social media manager, and you excel at creating viral and"
                                          " highly engaging posts for X (formerly Twitter)."},
            {"role": "user", "content": f"{prompt}"},
        ],
        temperature=1.5,
        stream=False
    )

    return response.choices[0].message.content


def main():
    print("Welcome to the Advanced AI X Post Generator!")
    website_url = input("Please enter a valid Website URL to fetch and summarize: ")
    print("Fetching website HTML...")
    try:
        html_content = get_website_html(website_url)
    except Exception as e:
        print(f"An error occurred while fetching the website: {e}")
        return

    if not html_content:
        print("Failed to fetch the website content. Exiting.")
        return

    print("---------")
    print("Extracting core content from the website...")
    core_content = extract_core_website_content(html_content)
    print("Extracted core content:")
    print(core_content)

    print("---------")
    print("Summarizing the core content...")
    summary = summarize_content(core_content)
    print("Generated summary:")
    print(summary)

    print("---------")
    print("Generating X post based on the summary...")
    x_post = generate_x_post(summary)
    print("Generated X post:")
    print(x_post)


if __name__ == "__main__":
    main()