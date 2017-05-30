from matplotlib import pyplot as plt
import numpy as np
import os.path

def running_mean(x, N):
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[N:] - cumsum[:-N]) / N 


def get_best_loss(dir_list, win_width):
  loss_dict = dict()
  best_loss_last = 1e10
  best_file = ""
  for dir in dir_list:
    if os.path.isfile(dir + "/loss_full.txt"):
      file_name = dir + "/loss_full.txt"
    elif os.path.isfile(dir + "/loss.txt"):
      file_name = dir + "/loss.txt"
    else:
      raise Exception("loss file not found")

    loss = np.loadtxt(file_name)
    loss = running_mean(loss, win_width)
    loss_dict[file_name] = loss
    print file_name, " loss length ", loss.size
    if np.min(loss) < best_loss_last:
      best_loss = loss
      best_loss_last = np.min(loss)
      best_file = file_name

  return best_loss, best_file, loss_dict


def get_loss(dir, win_width):
  if os.path.isfile(dir + "/loss_full.txt"):
    file_name = dir + "/loss_full.txt"
  elif os.path.isfile(dir + "/loss.txt"):
    file_name = dir + "/loss.txt"
  else:
    raise Exception("loss file not found")

  loss = np.loadtxt(file_name)
  loss = running_mean(loss, win_width)
  return loss


def get_best_steps_to_target_loss(dir_list, win_width, target_loss):
  steps = dict()
  best_step = None
  for dir in dir_list:
    loss = get_loss(dir, win_width)
    if np.min(loss) > target_loss:
        steps[dir] = None
    else:
        steps[dir] = np.argmax(loss < target_loss)
        
    if best_step == None:
        best_step = steps[dir]
    elif steps[dir] != None and best_step > steps[dir]:
        best_step = steps[dir]
  return best_step, steps


# def running_mean(x, N):
#   cumsum = np.cumsum(np.insert(x, 0, 0)) 
#   return (cumsum[N:] - cumsum[:-N]) / N 


# def get_best_loss(dir_list, win_width):
#   loss_dict = dict()
#   best_loss_last = 1e10
#   best_file = ""
#   for dir in dir_list:
#     if os.path.isfile(dir + "/loss_full.txt"):
#       file_name = dir + "/loss_full.txt"
#     elif os.path.isfile(dir + "/loss.txt"):
#       file_name = dir + "/loss.txt"
#     else:
#       raise Exception("loss file not found")

#     loss = np.loadtxt(file_name)
#     loss = running_mean(loss, 20)
#     loss_dict[file_name] = loss
#     if loss[-1] < best_loss_last:
#       best_loss = loss
#       best_loss_last = loss[-1]
#       best_file = file_name

#   return best_loss, best_file, loss_dict


# def get_loss(dir):
#   if os.path.isfile(dir + "/loss_full.txt"):
#     file_name = dir + "/loss_full.txt"
#   elif os.path.isfile(dir + "/loss.txt"):
#     file_name = dir + "/loss.txt"
#   else:
#     raise Exception("loss file not found")

#   loss = np.loadtxt(file_name)
#   loss = running_mean(loss, 20)
#   return loss


