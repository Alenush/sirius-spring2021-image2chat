def filter_bad(data, hashes_set=None):
  # filters bad samples from dataset
  new_data = []
  for elem in data:
    bad_flag = False
    if hashes_set is not None and elem['image_hash'] not in hashes_set:
      continue
    for utt in elem['dialog']:
      if '[' in utt[1] or utt[0] == 'Crude' or utt[0] == 'Earnest':
        bad_flag = True
        break
    if not bad_flag:
      new_data.append(elem)
  return new_data
