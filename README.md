!Don't move any files!

On Windows
To create and train models (unnecessary): 
>>python sp.py

For spambot functionality on Discord:
>>python api.py
>>python bot.py

### This is how I do it. However, for the Discord bot, it requires you to make your own application in Discord (https://discord.com/developers/applications?new_application=true) unless you just want to run my bot on my server (On the off-chance that you want to do this let me know your discord ID and I can invite you). 

When you create your own bot, it comes with its own token as well as permissions. For this application I believe permissions must be given to be able to at least read messages (in the OAuth2 and Bot tab). I just gave it as many permissions as I could so I might be wrong! You also need a token from the Bot tab. This is what you will replace the DISCORD_BOT_TOKEN static variable's value in bot.py for it to work with your own bot. Then, add the bot to your own server and you can test it out by sending messages for it to classify. If everything goes correctly it should output a message making a prediction.

For commands within Discord:
"yes [ham/spam/phishing/benign]"
"no [ham/spam/phishing/benign]"
"!update"

ex.
typing "yes ham" will add the previous message that was just predicted upon to a feedback file "newspam.csv"
yes or no doesn't matter.
!update will retrain the models on the newspam.csv and newphish.csv data points.