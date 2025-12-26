import asyncio
import websockets
import numpy as np
from faster_whisper import WhisperModel
from googletrans import Translator
import os

# إعداد المحرك
print("--- بدء تشغيل السيرفر (نسخة الترجمة العربية) ---")
model = WhisperModel("base", device="cpu", compute_type="int8")
translator = Translator()
print("الموديل جاهز ومستعد لاستقبال الصوت...")

async def audio_processor(websocket):
    print(f"تم اتصال جديد من: {websocket.remote_address}")
    audio_buffer = bytearray()

    try:
        async for message in websocket:
            audio_buffer.extend(message)

            # معالجة كل 4 ثوانٍ تقريباً
            if len(audio_buffer) >= 128000:
                audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                
                # تحويل الصوت لنص إنجليزي
                segments, _ = model.transcribe(audio_np, beam_size=1)
                
                for segment in segments:
                    eng_text = segment.text.strip()
                    if eng_text and len(eng_text) > 2:
                        print(f"قيل: {eng_text}")
                        try:
                            # الترجمة للعربية
                            translation = translator.translate(eng_text, src='en', dest='ar')
                            arabic_text = translation.text
                            print(f"الترجمة: {arabic_text}")
                            
                            # إرسال النص العربي
                            await websocket.send(arabic_text)
                        except Exception as e:
                            print(f"خطأ في الترجمة: {e}")
                            await websocket.send(eng_text)
                
                audio_buffer.clear()

    except Exception as e:
        print(f"انقطع الاتصال: {e}")

async def main():
    # قراءة البورت من متغيرات البيئة (مهم للنشر أونلاين)
    port = int(os.environ.get("PORT", 8000))
    
    async with websockets.serve(audio_processor, "0.0.0.0", port):
        print(f"السيرفر شغال على البورت {port}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
