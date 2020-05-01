from threading import Thread

def start_daemon(target):
  t = Thread(target=target)
  t.daemon = True
  t.start()

class Starter():
  def __init__(self, runnable):
    self.runnable = runnable
  def update(self, obs, msg):
    self.runnable.run()

class Stopper:
  def __init__(self, runnable):
    self.runnable = runnable
  def update(self, observable, message):
    self.runnable.running = False