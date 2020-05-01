DIR = 'D:\\Projects\\background-data\\public\\data\\'
def quick_write(file, content):
  with open(f'{DIR}{file}', 'w+') as f:
    f.write(content)
def quick_read(file):
  with open(f'{DIR}{file}', 'r') as f:
    return f.read()