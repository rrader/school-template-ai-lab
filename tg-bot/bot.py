import telepot
from telepot.loop import MessageLoop
from openai import OpenAI
import os


client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.environ["OPENROUTER_KEY"],
)


def message(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)

    print(content_type, chat_type, chat_id)

    completion = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
          {
                "role": "system",
                "content": "Ти - бот що відповідає із гумором на запитання. "
            },
            {
                "role": "user",
                "content": msg['text']
            }
        ]
    )

    bot.sendMessage(chat_id, completion.choices[0].message.content)


bot = telepot.Bot(os.environ["TG_KEY"])
MessageLoop(bot, {'chat': message}).run_forever()
