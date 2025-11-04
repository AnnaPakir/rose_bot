from fastapi import FastAPI, Request, HTTPException
import os
import logging
from telegram import Bot, Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import torch
from PIL import Image
import io
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from huggingface_hub import hf_hub_download

# === Логирование ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Telegram Diffusion Bot")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO")
device = "cuda" if torch.cuda.is_available() else "cpu"

# === DDPMScheduler и модель ===
class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.num_train_timesteps = num_train_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprod)

    def add_noise(self, original, noise, timesteps):
        timesteps = timesteps.to(self.device)
        sqrt_alpha_cumprod = self.sqrt_alpha_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod[timesteps].view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod * original + sqrt_one_minus_alpha_cumprod * noise

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original = (sample - self.sqrt_one_minus_alpha_cumprod[t] * model_output) / self.sqrt_alpha_cumprod[t]
        pred_original = torch.clamp(pred_original, -1.0, 1.0)
        mean = (sample - self.betas[t] * model_output / self.sqrt_one_minus_alpha_cumprod[t]) / torch.sqrt(self.alphas[t])
        if t > 0:
            noise = torch.randn_like(sample)
            variance = (1 - self.alpha_cumprod[t-1]) / (1 - self.alpha_cumprod[t]) * self.betas[t]
            sample = mean + torch.sqrt(variance) * noise
        else:
            sample = mean
        return sample

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)
        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        else:
            self.time_mlp = None
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        if self.time_mlp is not None and t_emb is not None:
            time_emb = self.time_mlp(t_emb)[:, :, None, None]
            h = h + time_emb
        return F.relu(h + self.shortcut(x))

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        time_emb_dim = 32
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        self.conv1 = Block(3, 64, time_emb_dim)
        self.down1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv2 = Block(64, 128, time_emb_dim)
        self.down2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.conv3 = Block(128, 256, time_emb_dim)
        self.down3 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.mid_conv1 = Block(256, 256, time_emb_dim)
        self.mid_conv2 = Block(256, 256, time_emb_dim)
        self.up1 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)
        self.conv4 = Block(512, 128, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.conv5 = Block(256, 64, time_emb_dim)
        self.up3 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.conv6 = Block(128, 64, time_emb_dim)
        self.final_conv = nn.Conv2d(64, 3, 1)

    def forward(self, x, timesteps):
        t_emb = self.time_mlp(timesteps)
        x1 = self.conv1(x, t_emb)
        x1d = self.down1(x1)
        x2 = self.conv2(x1d, t_emb)
        x2d = self.down2(x2)
        x3 = self.conv3(x2d, t_emb)
        x3d = self.down3(x3)
        x = self.mid_conv1(x3d, t_emb)
        x = self.mid_conv2(x, t_emb)
        x = self.up1(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv4(x, t_emb)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv5(x, t_emb)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv6(x, t_emb)
        return self.final_conv(x)

# === Инициализация модели ===
model = SimpleUNet().to(device)
scheduler = DDPMScheduler(device=device)

try:
    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename="diffusion_final.pth"
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info("Модель успешно загружена с Hugging Face Hub")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")

bot = None
application = None

async def initialize_bot():
    """Инициализация Telegram-бота"""
    global bot, application

    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN не установлен!")
        return False

    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        application = Application.builder().token(TELEGRAM_TOKEN).build()

        async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
            user_message = update.message.text
            await update.message.chat.send_action(action="typing")
            try:
                logger.info(f"Генерация изображения для: {user_message}")
                model.eval()
                with torch.no_grad():
                    sample = torch.randn(1, 3, 64, 64, device=device)
                    for t in range(999, -1, -1):
                        noise_pred = model(sample, torch.tensor([t], device=device))
                        sample = scheduler.step(noise_pred, t, sample)
                    image = sample.squeeze(0).cpu()
                    image = (image + 1) / 2
                    image = torch.clamp(image, 0, 1)
                    image = image.permute(1, 2, 0)
                    image_np = (image.numpy() * 255).astype(np.uint8)
                    generated_image = Image.fromarray(image_np)
                bio = io.BytesIO()
                generated_image.save(bio, format='PNG')
                bio.seek(0)
                await update.message.reply_photo(photo=bio, caption=f"🎨 {user_message}")
            except Exception as e:
                logger.error(f"Ошибка генерации: {e}")
                await update.message.reply_text("😞 Ошибка генерации изображения")

        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        await application.initialize()
        await application.start()
        logger.info("Бот успешно инициализирован")
        return True
    except Exception as e:
        logger.error(f"Ошибка инициализации бота: {e}")
        return False

bot_initialized = False

@app.post("/webhook")
async def webhook(request: Request):
    """Эндпоинт для Telegram webhook"""
    global bot_initialized
    if not bot_initialized:
        logger.info("Первая инициализация бота...")
        bot_initialized = await initialize_bot()
    if not bot_initialized:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    try:
        data = await request.json()
        update = Update.de_json(data, application.bot)
        await application.process_update(update)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Ошибка обработки вебхука: {e}")
        raise HTTPException(status_code=400, detail="Error processing update")

@app.get("/")
async def root():
    return {"status": "Диффузионный бот работает! Отправьте сообщение боту в Telegram."}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.on_event("startup")
async def on_startup():
    logger.info("Приложение запущено")
