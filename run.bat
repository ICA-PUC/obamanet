del input.mp4
del output.mp4

python run.py --sf %1
python pix2pix.py --mode test --output_dir data/output_images/ --input_dir data/image_lips/ --checkpoint checkpoints/output/

ffmpeg -y -r 30 -f image2 -s 256x256 -i data/output_images/images/%%d-outputs.png -vcodec mpeg4 -crf 25 output0.mp4
ffmpeg -y -r 30 -f image2 -s 256x256 -i data/output_images/images/%%d-inputs.png -vcodec mpeg4 -crf 25 input0.mp4
ffmpeg -y -i %1 output_audio_trim.wav

ffmpeg -y -i output0.mp4 -i output_audio_trim.wav -c:v copy -c:a aac -strict experimental output.mp4
ffmpeg -y -i input0.mp4 -i output_audio_trim.wav -c:v copy -c:a aac -strict experimental input.mp4

del /Q testing_output_images
del /Q data/output_images
del output0.mp4
del input0.mp4
del output_audio_trim.wav