import os
content = open('scripts/train/train_h_jepa.py').read()

import re
# Clean up duplicate one-hots
content = re.sub(r'a_seq_batch = F\.one_hot\(a_seq_batch\.to\(device\), num_classes=4\)\.float\(\)\n', 'a_seq_batch = a_seq_batch.to(device)\n', content)
content = re.sub(r'a_seq_onehot = F\.one_hot\(a_seq_batch\.long\(\), num_classes=4\)\.float\(\)\n', 'a_seq_onehot = F.one_hot(a_seq_batch.long(), num_classes=4).float()\n', content)

# Make sure the try/except is removed so we actually see errors!
content = content.replace('            if True:', '            try:')
content = content.replace('            if False:', '            except ValueError:')
# Wait, I want to SEE errors!
content = content.replace('            try:', '            if True:')
content = content.replace('            except ValueError:', '            if False:')

with open('scripts/train/train_h_jepa.py', 'w') as f:
    f.write(content)
