print('static const size_t nodeSizes[3][6] = {')
for type in ['float', 'Vec3f', 'int']:
  print('{', end='')
  for ld in range(2, 8):
    print(f'sizeof(InternalNode<LeafNode<{type}>, {ld}>)', end='')
    if ld != 7:
      print(', ', end='')
  if(type == 'int'):
    print('}')
  else:
    print('},')
print('};')

print('static const size_t leafSizes[3][6] = {')
for type in ['float', 'Vec3f', 'int']:
  print('{', end='')
  for ld in range(2, 8):
    print(f'sizeof(LeafNode<{type}, Coord, Mask, {ld}>)', end='')
    if ld != 7:
      print(', ', end='')
  if(type == 'int'):
    print('}')
  else:
    print('},')
print('};')

print('static const Node2RangeFunc rangeFunctions[3][6] = {')
for type in ['float', 'Vec3f', 'int']:
  print('{', end='')
  for ld in range(2, 8):
    print(f'GetNode2Range<{type}, {ld}>', end='')
    if ld != 7:
      print(', ', end='')
  if(type == 'int'):
    print('}')
  else:
    print('},')
print('};')

print('static const __device__ ProcessLeafFunc processLeafFuncs[3][6] = {')
for type in ['float', 'nanovdb::Vec3f', 'int']:
	print('{', end='')
	for ld in range(2, 8):
		print(f'ProcessLeaf<{type}, {ld}>', end='')
		if ld != 7:
			print(', ', end='')
	if(type == 'int'):
		print('}')
	else:
		print('},')
print('};')

print('static const __device__ NodeRangeFunc rangeFunctions[2][3][6] = {{')
for type in ['float', 'nanovdb::Vec3f', 'int']:
	print('{', end='')
	for ld in range(2, 8):
		print(f'GetNodeRange<LeafNodeSmpl<{type}, {ld}>>', end='')
		if ld != 7:
			print(', ', end='')
	if(type == 'int'):
		print('}')
	else:
		print('},')
print('},{',)
for type in ['float', 'nanovdb::Vec3f', 'int']:
	print('{', end='')
	for ld in range(2, 8):
    # Using 3 for the leaf node's LOG2DIM suffices here
		print(f'GetNodeRange<INodeSmpl<{type}, {ld}>>', end='')
		if ld != 7:
			print(', ', end='')
	if(type == 'int'):
		print('}')
	else:
		print('},')
print('}};')

print('static const __device__ ProcessInternalNodeFunc processInternalNodeFuncs[3][6] = {')
for type in ['float', 'nanovdb::Vec3f', 'int']:
	print('{', end='')
	for ld in range(2, 8):
		print(f'ProcessInternalNode<{type}, {ld}>', end='')
		if ld != 7:
			print(', ', end='')
	if(type == 'int'):
		print('}')
	else:
		print('},')
print('};')