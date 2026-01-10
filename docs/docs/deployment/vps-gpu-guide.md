# Deploy OM1 on VPS GPU Providers

This guide covers deploying OM1 on cloud GPU providers, with detailed walkthrough using OctaSpace as the primary example.

---

## Why Use Cloud GPU Providers?

- ✅ **No expensive hardware needed** - Rent GPU by the hour
- ✅ **Powerful GPUs** - Access to RTX 4090, A100, H100
- ✅ **Pay-as-you-go** - Only pay for what you use
- ✅ **Quick setup** - Deploy in minutes, not hours
- ✅ **Web-based access** - No complex SSH setup required

---

## Why OctaSpace?

I chose OctaSpace because:
- 💰 **Cheapest option** - ~$0.10/hr for RTX 4090
- 🌐 **Web terminal** - Use directly in browser
- ⚡ **Fast deployment** - Only 2-3 minutes

---

## Step 1: Choose GPU Instance

1. Sign up at [OctaSpace](https://marketplace.octa.space/)
2. Go to **Applications**
3. Search: **"ubuntu"**
4. **Select: Ubuntu 22.04 LTS** (DO NOT choose VM version)

### Choose suitable GPU:

Recommended: **RTX 4090 24GB** with:
- ✅ RAM ≥ 47GB
- ✅ Uptime score ≥ 90
- ✅ Network speed ≥ 250 Mbit
- ✅ Price: ~$0.10-0.11/hr

---

## Step 2: Configure Deployment

### Config settings:
```yaml
Selected app: Ubuntu 22.04 LTS
Start command: [leave empty]
Image name: ubuntu:22.04
Disk size: 50 GB
Expose HTTP Ports: [leave empty]
Expose TCP/UDP Ports: [leave empty]
Environment variables: [leave empty]
```

**Note:**
- 50GB disk is necessary for dependencies
- Leave HTTP Ports empty since we use web terminal

### Click the pink **"Deploy"** button!

⏰ **Wait 2-3 minutes** for setup

---

## Step 3: Open Web Terminal

✅ **No SSH client needed, no key setup required!**

You'll see the prompt:
```
root@[instance-id]:~#
```

---

## Step 4: Verify GPU

In web terminal, type:
```bash
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.x   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
|  0%   45C    P8    25W / 450W |      0MiB / 24576MiB |      0%      Default |
+-----------------------------------------------------------------------------+
```

✅ Seeing RTX 4090 24GB means you're good to go!

---

## Step 5: Install Dependencies

Copy-paste (use right-click to paste) each command block into terminal:

### Update system:
```bash
apt update && apt upgrade -y
```

### Install Python and tools:
```bash
apt install -y python3 python3-pip git curl wget vim build-essential
```

### Install audio libraries (for OM1):
```bash
apt install -y portaudio19-dev python3-dev gcc g++ make ffmpeg
```

### Install UV package manager:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Load UV into PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Verify UV:
```bash
uv --version
```

Expected output: `uv 0.x.xx`

### Install nano:
```bash
apt install -y nano
```

---

## Step 6: Clone Your Fork

**Important:** Clone **YOUR FORK**, not the original repo!
```bash
# Navigate to home
cd ~

# Clone your fork (replace YOUR_USERNAME)
git clone https://github.com/YOUR_USERNAME/OM1.git

# Example with username "windlgrass":
git clone https://github.com/windlgrass/OM1.git

# Enter directory
cd OM1
```

### Setup upstream (original repo):
```bash
git remote add upstream https://github.com/OpenMind/OM1.git
git remote -v
```

Expected output:
```
origin    https://github.com/YOUR_USERNAME/OM1.git (fetch)
origin    https://github.com/YOUR_USERNAME/OM1.git (push)
upstream  https://github.com/OpenMind/OM1.git (fetch)
upstream  https://github.com/OpenMind/OM1.git (push)
```

### Update submodules:
```bash
git submodule update --init
```

---

## Step 7: Setup Virtual Environment

### Create venv:
```bash
~/.local/bin/uv venv
```

If asked "replace .venv?", type: `y`

### Activate venv:
```bash
source .venv/bin/activate
```

Prompt will change to:
```
(om1) root@[instance-id]:~/OM1#
```

### Install dependencies:
```bash
~/.local/bin/uv sync
```

⏰ **Wait 5-10 minutes** for downloading and building packages.

**If you encounter errors:** Scroll past warnings, wait until completion.

---

## Step 8: Configure API Key

### Get API Key:

1. Visit [OpenMind Portal](https://portal.openmind.org)
2. Sign up/Login
3. Copy your API key

### Add API key to config:
```bash
# Edit config file (using nano)
nano config/spot.json5
```

**In nano:**
1. Use arrow keys to navigate to line: "api_key": "openmind_free"
2. Delete "openmind_free" (use Backspace or Delete)
3. Copy API key from OpenMind Portal and Paste into nano
4. Save: Press Ctrl+O, then press Enter
5. Exit: Press Ctrl+X
```bash
# Verify the change
cat config/spot.json5 | grep api_key
```

---

## Step 9: Run Spot Agent
```bash
# Run agent (with venv activated)
uv run src/run.py spot
```

### Successful test shows:
```
SPEAK: "Ooh, what was that?..."
EMOTION: excited
move: 'wag tail'
```

Example logs:
```
2026-01-08 03:44:25 - INFO - HTTP Request: POST https://api.openmind.org/...
2026-01-08 03:44:25 - INFO - Function calls: [...]
2026-01-08 03:44:25 - INFO - SendThisToROS2: {'speak': '...'}
```

---

## Step 10: Interact with Agent

You can now interact with Spot through the terminal. The agent will respond with emotions, movements, and speech.

---

## Step 11: Stop and Cleanup

### Stop agent:
```bash
# Press Ctrl+C
```

### Deactivate venv:
```bash
deactivate
```

### Terminate instance:

**IMPORTANT:** To avoid charges!

1. Return to OctaSpace dashboard
2. Find your running instance
3. Click **"Terminate"** or **"Stop"**
4. Confirm termination

✅ **Instance stopped, no more charges!**

---

## Community & Support

- **OM1 GitHub:** https://github.com/OpenMind/OM1
- **OpenMind Portal:** https://portal.openmind.org
- **Documentation:** https://docs.openmind.org
- **Discord:** https://discord.gg/openmind
- **X (Twitter):** https://x.com/openmind_agi
- **Issues:** Report bugs on GitHub Issues

---

## Credits

*Guide created by [@windlgrass](https://github.com/windlgrass) - OM1 Community Contributor*

*If this guide helped you, give it a ⭐ and share with friends!*

**Version:** 1.0  
**Last Updated:** January 2026  
**Tested on:** OctaSpace RTX 4090 instances

---

## License

This documentation is part of the OM1 project and follows the same license terms.
