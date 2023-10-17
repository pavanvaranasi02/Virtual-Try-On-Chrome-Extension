# Virtual-Try-On-Chrome-Extension
Developed a Chrome Extension enabling users to virtually try on clothing and accessories by superimposing input images onto the user's photo, enhancing online shopping experiences through ml technology.

Complete this mentioned steps before running the application.py flask app:
1. Create a folder named checkpoints and save GMM.pth, TOM.pth, inference.pth, Openpose model
(You can download GMM and TOM from this website: "https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP")
(inference.pth file in "https://drive.google.com/uc?id=1eUe18HoH05p0yFUd_sN6GXdTj82aW0m9")
(Open Pose model from kaggle in "https://www.kaggle.com/datasets/changethetuneman/openpose-model?select=pose_iter_440000.caffemodel")

2. pip install -r requirements.txt, install all required python modules.
3. then create a result folder in static folder.
4. If you want web application, directly execute application.py
5. for chrome extension, you need to open chrome go to "chrome://extensions" and enable developer mode and click load unpacked and open or drag and drop chrome extension folder.

# Note:
6. then your extension is loaded into chrome and important note is that, even if it is a chrome extension running application.py in background is mandatory.
7. Finally you can create key.pem and cert.pem to open your website in a more secured way that is open your website using "https".
8. For now your good to run all viton dataset images on it, if you upload your own images it would not produce best results.

viton dataset is available at: "https://1drv.ms/u/s!Ai8t8GAHdzVUiQRFmTPrtrAy0ZP5?e=rS1aK8".

And finally for chrome extesnsion, you need to change the file path of input and result image in response json in application.py file.

if you have any issues other than mentioned, feel free to post it on issues section.

