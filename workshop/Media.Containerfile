FROM sha256:f1a63cbdd0c4afbe35c533efa8f3ca21ff4f46c80a36d4d9cbba5bca08826f6c

# Operator provisioning only. Runtime containers never access this network.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg=7:7.1.5-0+deb13u1 \
    && dpkg-query -W > /opt/open-lab-workshop/media-packages.txt \
    && ffmpeg -version > /opt/open-lab-workshop/ffmpeg-version.txt \
    && ffmpeg -buildconf > /opt/open-lab-workshop/ffmpeg-buildconf.txt 2>&1 \
    && rm -rf /var/lib/apt/lists/*
