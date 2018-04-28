downloa
#install slack-cleaner
#install pip first

sudo easy_install pip
sudo pip install slack-cleaner

#get a token
https://api.slack.com/custom-integrations/legacy-tokens

#read instructions
slack-cleaner --help

usage: slack-cleaner [-h] --token TOKEN [--log] [--rate RATE]
                     (--message | --file)
                     [--channel CHANNEL | --direct DIRECT | --group GROUP | --mpdirect MPDIRECT]
                     [--user USER] [--bot] [--after AFTER] [--before BEFORE]
                     [--types TYPES] [--perform]

optional arguments:
  -h, --help           show this help message and exit
  --token TOKEN        Slack API token (https://api.slack.com/web)
  --log                Create a log file in the current directory
  --rate RATE          Delay between API calls (in seconds)
  --message            Delete messages
  --file               Delete files
  --channel CHANNEL    Channel name's, e.g., general
  --direct DIRECT      Direct message's name, e.g., sherry
  --group GROUP        Private group's name
  --mpdirect MPDIRECT  Multiparty direct message's name, e.g.,
                       sherry,james,johndoe
  --user USER          Delete messages/files from certain user
  --bot                Delete messages from bots

#how i run the command
slack-cleaner --token {token} --message --direct {colleague} --user {user} --perform
