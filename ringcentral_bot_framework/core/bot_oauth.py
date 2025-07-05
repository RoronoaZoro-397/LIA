"""
bot auth
"""
import time
from .common import result, debug, getQueryParam
from pydash.predicates import is_number, is_string

def initBotAuthHandler(conf, Bot, dbAction):
  def botAuth(event):
    bot = Bot()
    code = getQueryParam(event, 'code')
    auth_success = False
    if is_string(code):
      auth_success = bot.auth(code)
    else:
      auth_success = bot.authPrivateBot(event['body'])

    if not auth_success:
      return result('Bot authentication failed - check network connectivity and credentials', 500)

    bot.renewWebHooks(event)
    conf.botAuthAction(bot, dbAction)
    return result('Bot added')

  def renewBot (event):
    """
    for self call renewbot async
    """
    debug('self tringgering renew bot')
    if is_number(event['wait']):
      time.sleep(event['wait'])

    bot = Bot()
    bot.id = event['botId']
    bot.token = event['token']
    bot.rc.token = bot.token
    bot.writeToDb({
      'id': bot.id,
      'token': bot.token,
      'data': bot.data
    })
    bot.renewWebHooks(event)
    return result('Bot renew done')

  return botAuth, renewBot
