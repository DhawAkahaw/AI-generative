import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
from authtoken import auth_token
import torch
from diffusers import StableDiffusionPipeline
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")
prompt = ctk.CTkEntry(master=app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)
lmain = ctk.CTkLabel(master=app, height=512, width=512, text="")  # Add a blank text to suppress warnings
lmain.place(x=10, y=110)
status_label = ctk.CTkLabel(master=app, text="", font=("Arial", 16), text_color="red")
status_label.place(x=10, y=630)
modelid = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    modelid, variant="fp16", torch_dtype=torch.float32, use_auth_token=auth_token
)
pipe.enable_sequential_cpu_offload()
pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))  
from customtkinter import CTkImage
def generate():
    try:
        status_label.configure(text="Generating... Please wait.")
        app.update_idletasks()  
        result = pipe(prompt.get(), guidance_scale=8.5)
        image = result.images[0]  
        # Save the image
        image.save('generatedimage.png')
        ctk_image = CTkImage(light_image=image, dark_image=image, size=(512, 512))
        lmain.configure(image=ctk_image)
        status_label.configure(text="Image generated successfully!")
    except Exception as e:
        status_label.configure(text=f"An error occurred: {e}")
        print(f"An error occurred: {e}")
trigger = ctk.CTkButton(master=app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
