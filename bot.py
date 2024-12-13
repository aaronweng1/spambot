import discord
import aiohttp  # For async HTTP requests
import asyncio
import logging
import re
import csv
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)

API_URL = "http://127.0.0.1:5000/classify"
DISCORD_BOT_TOKEN = 'Custom Token'
# Define paths to the CSV files
NEWSPAM_FILE_PATH = 'newspam.csv'
NEWPHISH_FILE_PATH = 'newphish.csv'

intents = discord.Intents.default()
client = discord.Client(intents=intents)
last_classification = {}

@client.event
async def on_ready():
    logging.info(f'Logged in as {client.user}')

@client.event
async def on_message(message):
    global last_classification

    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    # Function to check if a message contains a URL
    def is_url(text):
        return bool(re.search(r'http[s]?://|www\.|\.com|\.org|\.net|\.co|\.edu|\.gov|\.info|\.tv|\.me|\.wiki|\.in|\.eu|\.asia|\.xyz|\.ly|\.site', text))
    
    # Check the previous message for feedback
    async for msg in message.channel.history(limit=2):
        if msg.author != client.user:
            print(msg.content.lower())  # Debugging: Check what the message content is
            
            # Check for "yes" or "no" responses
            if msg.content.lower().startswith("yes") or msg.content.lower().startswith("no"):
                # Extract the classification (after the "yes" or "no")
                parts = msg.content.lower().split(maxsplit=1)
                
                if len(parts) > 1:
                    feedback = parts[1]  # The part after "yes" or "no"
                else:
                    continue  # Skip if no classification part is found

                # Check if the user has previously classified a message
                if msg.author.id in last_classification:
                    last_message, result, last_url = last_classification[msg.author.id]

                    # Check feedback and decide where to append the message
                    if feedback in ['ham', 'spam']:
                        # Append to newspam.csv with <spam/ham> <tab> <message>
                        with open(NEWSPAM_FILE_PATH, mode='a', newline='', encoding='utf-8') as file:
                            writer = csv.writer(file, delimiter='\t')
                            for x in range (100):
                                writer.writerow([feedback, last_message])  # Store feedback and message
                            logging.info(f"Appended to {NEWSPAM_FILE_PATH}: {feedback}\t{last_message}")
                    else:
                        # Append to newphish.csv with <message>, <label>, <url>
                        with open(NEWPHISH_FILE_PATH, mode='a', newline='', encoding='utf-8') as file:
                            writer = csv.writer(file)
                            for x in range (100):
                                writer.writerow([last_url, feedback])  # Store message, feedback, and URL
                            logging.info(f"Appended to {NEWPHISH_FILE_PATH}: {last_url}, {feedback}")
                    
                    await message.channel.send("Thank you for your feedback!")
                else:
                    await message.channel.send("No recent message to provide feedback for.")
                return  # Skip the rest of the code (do not classify the "yes"/"no" message)
            
            if msg.content.lower().startswith("!update"):
                await message.channel.send("Starting model retraining with feedback...")

                try:
                    from sp import fine_tune_rnn, update_url_classifier
                    update_url_classifier()
                    fine_tune_rnn()
                    await message.channel.send("Model retrained successfully with feedback!")
                except Exception as e:
                    await message.channel.send(f"Error during retraining: {str(e)}")
                return  # Stop further processing of the message

    # Perform classification for the user's message (if not a feedback response)
    async for msg in message.channel.history(limit=2):
        if msg.author != client.user:
            is_message_url = is_url(msg.content)
            payload = {"url": msg.content} if is_message_url else {"text": msg.content}
            break

    # Asynchronously send request to API
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(API_URL, json=payload) as response:
                if response.status == 200:  # Use response.status instead of status_code
                    result = await response.json()

                    # Store classification for feedback reference
                    last_message = msg.content
                    last_url = msg.content if is_message_url else None
                    last_classification[message.author.id] = (last_message, result, last_url)

                    # Respond with classification result
                    if 'combined_prediction' in result:
                        await message.channel.send(f"⚠️ Classification: {result['combined_prediction']}")
                    elif 'spam_prediction' in result:
                        await message.channel.send(f"⚠️ Classification: {result['spam_prediction']}")
                    elif 'url_prediction' in result:
                        await message.channel.send(f"⚠️ Classification: {result['url_prediction']}")
                    else:
                        await message.channel.send("❌ Unexpected response from the spam API.")
                else:
                    await message.channel.send(f"❌ Error contacting the spam API. Status: {response.status}")
        except Exception as e:
            await message.channel.send(f"❌ Error during API request: {str(e)}")


if __name__ == '__main__':
    client.run(DISCORD_BOT_TOKEN)
