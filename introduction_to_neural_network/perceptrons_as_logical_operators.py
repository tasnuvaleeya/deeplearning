import pandas as pd

# set weight, weight2, bias

weight1 = 1.0
weight2 = 1.0
bias = -1.50


linear_combination = weight1 * 0 + weight2 * 1 + bias
print(linear_combination)

test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]
outputs = []
# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    # print(test_input[1])
    # print("------------")
    # print(test_input[1])
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias

    # print(test_input[0])
    # print("-----------------")
    # print(test_input[1])
    output = int(linear_combination >= 0)
    # print("output ", output)
    # print(output)
    is_correct_string ='Yes' if output == correct_output else 'No'
    # print("is_correct", is_correct_string)
    # print("co ", correct_output)
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])
# print("output", outputs)
# Print output

num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', 'Input 2', 'Linear Combination', 'Activation Output', 'Is Correct'])

if not num_wrong:
    print('You got all correct!\n')
else:
    print('You got {} wrong!\n'.format(num_wrong))
print(output_frame.to_string(index=False))

# 103.17.69.119:80
