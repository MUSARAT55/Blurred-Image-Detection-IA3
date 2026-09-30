# Render Deployment

Create a Render Web Service from this repository.

Build Command:
`pip install -r requirements.txt`

Start Command:
`gunicorn app:app`

Python:
`3.13`

After deployment, Render gives a public HTTPS `.onrender.com` URL that can be opened on phones and other devices.

The trained CNN should be kept at:
`model/blur_detector.h5`

Do not upload passwords, API keys, or other secrets.
