import torch
# import struct
# def binary(num):
#     return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))

# # print("binary(1.0)", binary(1.0))
# # print("binary(-1.0)", binary(-1.0))
# print("binary(19.787)", binary(19.787))
# # print("binary(-0.346)", binary(-0.346))
# # print("binary(0.346)", binary(0.346))

# neuron_size = "3193620235260260380860395315386767256775104292050422229173980784746818233202936592714952417102972943691359030246581955571972669294782459642630729973041419806972391941076363199282936423274895013242103318753760717682495337762363243823612132696777532858130448743425525288791685658850689418121715997934424695016483808865200750337967286290071242945572682390280383534725510863926323884796972057669875406769131714691912681127936000"
# print(len(neuron_size))

a = torch.tensor([[1, 2], [5, 6]])
b = torch.tensor([[1, 3], [4, 4]])
# compare = torch.gt(a,b)

compare1 = torch.gt(a,2.0)
compare2 = torch.gt(a,1.0)


compare = compare1 * compare2

print(compare1)
print(compare2)
print(compare)

# a[compare == True] = b[compare == True]

print(a)
# float_to_bin = binary(-0.346)
# new_string = float_to_bin[:3] + '0' + float_to_bin[4:]
# print(new_string)

# # for reconversion
# f = int(new_string, 2)
# print(f)
# print(struct.unpack('f', struct.pack('I', f))[0])

