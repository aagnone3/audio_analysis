#!/bin/sh

download_this () {
    # Download data from a URL to a zip, and unzip to the desired directory
    # $1 link to data
    # $2 destination directory for unzipped data
    wget -O data.zip $1
    unzip data.zip -d $2
    rm data.zip
}

# create the data directories if necessary
echo "Ensuring directories exist"
mkdir -p ../res/features
mkdir -p ../res/models
mkdir -p ../res/results
mkdir -p ../res/data/speaker_recognition/
mkdir -p ../res/data/audio_classification/

# ELSDSR speaker recognition database http://www.imm.dtu.dk/~lfen/elsdsr/
SR_DIR="../res/data/speaker_recognition"
if ! [ "$(ls -A $SR_DIR)" ]; then
    # nothing in the speaker recognition directory, download the data into it
    echo "Downloading speaker recognition data, just a moment"
    download_this https://www.dropbox.com/sh/kwb9ro3ol9q5cyk/AAAr8E5iesiA5sNxxN0tw7bGa?dl=1 ../res/data/speaker_recognition/
fi

# Audio classification data from various sources TBD
#download_this 
