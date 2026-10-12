cur_x = 1
rate = 0.01  # change learning rate to see if it converges faster or slower
precision = 0.000001
previous_step_size = 1
max_iters = 10000
iters = 0
df = lambda x: 2 * (x - 6)

while previous_step_size > precision and iters < max_iters:
    prev_x = cur_x
    cur_x = cur_x - rate * df(prev_x)
    previous_step_size = abs(cur_x - prev_x)
    iters = iters + 1
    print("Iteration", iters, "\nX value is", cur_x)

print("The local minimum occurs at", cur_x)