"""Train the existing MLP and memory readout, with the full brain/GRU frozen.

Reconstruct recurrent memory from complete recorded observation histories and
verify its proposals against native GPU collection before permitting training.
There is no new inference module and no runtime oracle.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--steps', type=int, default=30000)
p.add_argument('--tag', default='readouts0')
p.add_argument('--lr', type=float, default=3e-5)
p.add_argument('--parent')
p.add_argument('--save-every', type=int, default=10000)
args = p.parse_args()
torch.manual_seed(20260917)
torch.set_num_threads(4)
assert torch.cuda.is_available() and '4080' in torch.cuda.get_device_name()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = 'cuda'
meta = json.loads((ROOT / 'model/model.json').read_text())
original = np.fromfile(ROOT / 'model/weights.bin', dtype=np.float32)
folder = ROOT / 'local-results/full-campaign-training'
out = folder / 'checkpoints'
out.mkdir(exist_ok=True)
registry = {meta['weights_sha256']: original}
for file in out.glob('*.json'):
    info = json.loads(file.read_text())
    if info.get('schema') == 'full-connectome-decoder-finetune-v1':
        binary = (file.parent / info['weightsFile']).read_bytes()
        assert hashlib.sha256(binary).hexdigest() == info['weightsSha256']
        registry[info['weightsSha256']] = np.frombuffer(binary, dtype=np.float32)
parent_info = json.loads(Path(args.parent).read_text()) if args.parent else None
parent_hash = parent_info['weightsSha256'] if parent_info else meta['weights_sha256']


def tensor(name, flat=original):
    a = meta['arrays'][name]
    return torch.tensor(flat[a['offset']:a['offset'] + a['length']].copy().reshape(a['shape']), device=device)


class Readouts(nn.Module):
    def __init__(self, flat):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 6))
        self.residual = nn.Linear(128, 6)
        self.load_state_dict({n: tensor(n, flat) for n in self.state_dict()})

    def forward(self, x, h):
        return self.net(x) + self.residual(h)


xmean, xstd, ymean, ystd = [tensor(n) for n in ('input_mean', 'input_std', 'target_mean', 'target_std')]
embed = nn.Linear(512, 128).to(device)
embed.load_state_dict({n: tensor('embed.0.' + n) for n in embed.state_dict()})
gru = nn.GRU(128, 128).to(device)
gru.load_state_dict({n: tensor('gru.' + n) for n in gru.state_dict()})
embed.eval(); gru.eval()
original_readout = Readouts(original).to(device).eval()
model = Readouts(registry[parent_hash]).to(device)
banks = {'train': [], 'validation': []}
manifest = []
max_parity_error = 0.
records = [(path, json.loads(path.read_text())) for path in sorted(folder.glob('r*-*.json'))]
records = [(path, r) for path, r in records if r.get('schema') == 'full-brain-decoder-dagger-v1']
relabelled_easy = {r['level'] for _, r in records if r['level'] <= 60 and r.get('labelAllSafe')}
with torch.no_grad():
    for path, r in records:
        # Supersede old easy-gate imitation once direct, safe-path labels are
        # available. Keeping both would demand two incompatible action targets.
        if r['level'] in relabelled_easy and not r.get('labelAllSafe'):
            continue
        binary = (folder / r['file']).read_bytes()
        assert hashlib.sha256(binary).hexdigest() == r['sha256']
        raw = torch.tensor(np.frombuffer(binary, dtype=np.float32).copy().reshape(-1, 8, 525), device=device)
        assert raw.shape[0] == r['ticks'] and r['fullConnectome']
        x = ((raw[:, :, :512] - xmean) / xstd).clamp(-10, 10)
        h, _ = gru(nn.functional.silu(embed(x)))
        x = x.reshape(-1, 512); h = h.reshape(-1, 128)
        raw = raw.reshape(-1, 525)
        ref = Readouts(registry[r['studentWeights']]).to(device).eval()
        replayed = ref(x, h) * ystd + ymean
        err = (replayed - raw[:, 518:524]).abs().max().item()
        max_parity_error = max(err, max_parity_error)
        assert err < 2e-4, (path.name, 'Recurrent reconstruction differs from native inference', err)
        # Always anchor non-teacher states to the ORIGINAL readout, including
        # candidate-induced histories. Do not recursively reinforce drift.
        baseline = original_readout(x, h) * ystd + ymean
        flag = raw[:, 524]
        target = torch.where(flag[:, None] > 0, raw[:, 512:518], baseline)
        banks[r['split']].append((x, h, target, flag, baseline))
        manifest.append({'file': path.name, 'sha256': r['sha256'], 'level': r['level'], 'split': r['split'], 'studentWeights': r['studentWeights']})
        del ref
data = {k: tuple(torch.cat([v[i] for v in b]) for i in range(5)) for k, b in banks.items()}
assert not ({r['level'] for r in manifest if r['split'] == 'train'} & {r['level'] for r in manifest if r['split'] == 'validation'})
x, h, target, flag, baseline = data['train']
previous_parts = []
offset = 0
for episode in banks['train']:
    count = len(episode[0]); index = torch.arange(count, device=device)
    previous_parts.append(torch.where(index >= 8, index - 8, index) + offset)
    offset += count
previous = torch.cat(previous_parts)
changed = torch.where(flag > 0)[0]; unchanged = torch.where(flag == 0)[0]
recovery = torch.where(flag > 1)[0]
limits = torch.tensor([.0144, .0144, .032], device=device)


def decode(raw):
    v = raw[:, :3] / limits
    ratio = torch.maximum(torch.linalg.vector_norm(v[:, :2], dim=1), v[:, 2].abs()).clamp_min(1)
    return v / ratio[:, None]


def loss_fn(raw, target, baseline):
    slow = torch.maximum(torch.linalg.vector_norm(target[:, :2] / .0144, dim=1), target[:, 2].abs() / .032) < .8
    action = (decode(raw) - decode(target)).square().mean(-1)
    braking = nn.functional.smooth_l1_loss(raw[:, :3] / limits, target[:, :3] / limits, reduction='none').mean(-1)
    return action + .1 * slow * braking + .02 * (raw[:, 3:] - target[:, 3:]).square().mean(-1) + 1e-5 * ((raw - baseline) / ystd).square().mean(-1)


optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
print(json.dumps({'start': True, 'trainRows': len(x), 'validationRows': len(data['validation'][0]), 'maxNativeParityError': max_parity_error, 'trainable': sum(t.numel() for t in model.parameters()), 'frozen': 'full connectome, projection, embedding, GRU recurrence', 'device': torch.cuda.get_device_name()}), flush=True)
start = time.time()
for step in range(1, args.steps + 1):
    indices = [changed[torch.randint(len(changed), (192 if len(recovery) else 256,), device=device)], unchanged[torch.randint(len(unchanged), (256,), device=device)]]
    if len(recovery): indices.append(recovery[torch.randint(len(recovery), (64,), device=device)])
    idx = torch.cat(indices)
    model.train()
    raw = model(x[idx], h[idx]) * ystd + ymean
    loss = (loss_fn(raw, target[idx], baseline[idx]) * torch.where(flag[idx] > 0, 1., 10.)).mean()
    # Match temporal changes, rather than suppressing necessary turns. Never
    # compare different agents or cross an episode reset boundary.
    prev = previous[idx]
    prev_raw = model(x[prev], h[prev]) * ystd + ymean
    desired_change = decode(target[idx]) - decode(target[prev])
    loss = loss + .05 * (decode(raw) - decode(prev_raw) - desired_change).square().mean()
    optim.zero_grad(set_to_none=True); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optim.step()
    if step % args.save_every == 0 or step == args.steps:
        model.eval()
        with torch.no_grad():
            vx, vh, vt, vf, vb = data['validation']
            errors = torch.cat([loss_fn(model(a, b) * ystd + ymean, c, d) for a, b, c, d in zip(vx.split(4096), vh.split(4096), vt.split(4096), vb.split(4096))])
        flat = original.copy(); frozen = np.ones(len(flat), dtype=bool)
        for name, value in model.state_dict().items():
            a = meta['arrays'][name]; region = slice(a['offset'], a['offset'] + a['length'])
            flat[region] = value.detach().cpu().numpy().reshape(-1); frozen[region] = False
        assert np.array_equal(flat[frozen], original[frozen])
        stem = f'{args.tag}-step-{step:05d}'
        binary = flat.tobytes(); (out / (stem + '.bin')).write_bytes(binary)
        torch.save({'readouts': model.state_dict(), 'parentWeights': parent_hash, 'data': manifest}, out / (stem + '.pt'))
        metrics = {'loss': loss.item(), 'validationTeacher': errors[vf > 0].mean().item(), 'validationPreserve': errors[vf == 0].mean().item(), 'seconds': time.time() - start}
        info = {'schema': 'full-connectome-decoder-finetune-v1', 'fullConnectome': True, 'weightsFile': stem + '.bin', 'weightsSha256': hashlib.sha256(binary).hexdigest(), 'checkpointSha256': hashlib.sha256((out / (stem + '.pt')).read_bytes()).hexdigest(), 'parentWeights': parent_hash, 'parentCheckpoint': parent_info['checkpointSha256'] if parent_info else meta['checkpoint_sha256'], 'step': step, 'metrics': metrics, 'data': manifest, 'frozenParametersBitExact': True, 'trainable': 'original net.* and residual.* readouts only', 'lossMode': 'post-limit actions + unsaturated braking', 'maxNativeParityError': max_parity_error, 'trainerSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (out / (stem + '.json')).write_text(json.dumps(info))
        print(json.dumps({'checkpoint': str(out / (stem + '.json')), **metrics}), flush=True)
