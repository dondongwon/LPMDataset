#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Google Cloud Speech API sample that demonstrates word time offsets.
Example usage:
    python transcribe_word_time_offsets.py resources/audio.raw
    python transcribe_word_time_offsets.py \
        gs://cloud-samples-tests/speech/vr.flac
"""
import argparse
import io
import os
import subprocess
import pandas as pd
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from google.cloud import storage
import time


parser = argparse.ArgumentParser()
parser.add_argument('-dir_path','--dir_path')
parser.add_argument('-bucket_name', '--bucket_name')
args = parser.parse_args()

DIR_PATH = args.dir_path     #Input directory with all audio
BUCKET_NAME = args.bucket_name



def upload_blob(bucket_name, source_file_name, destination_blob_name):
    #"""Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

def delete_blob(bucket_name, blob_name):
    #"""Deletes a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.delete()

def transcribe_file_with_word_time_offsets(audio_file_name, bucket_name):
    """Transcribe the given audio file synchronously and output the word time
    offsets."""
    start = time.time()

    client = speech.SpeechClient()

    #convert to flac, as desired by GOOGLE ASR
    #subprocess.call('ffmpeg -i stereo.flac -ac 1 mono.flac', shell=True
    #subprocess.call('ffmpeg -i stereo.flac -ac 1 mono.flac', shell=True)
    source_file_name = os.path.join(DIR_PATH, audio_file_name)
    destination_blob_name = audio_file_name
    print("\nuploading {} to google cloud bucket...\n".format(audio_file_name))
    upload_blob(bucket_name, source_file_name, destination_blob_name)
    gcs_uri = 'gs://' + bucket_name + '/' + destination_blob_name
    transcript_no_offset = ''
    transcript = ''
    alignment = ""

    client = speech.SpeechClient()
    audio = types.RecognitionAudio(uri=gcs_uri)
    config = types.RecognitionConfig(
    encoding=enums.RecognitionConfig.AudioEncoding.FLAC,
    sample_rate_hertz=44100,
    language_code='en-US',
    enable_word_time_offsets=True)
    operation = client.long_running_recognize(config, audio)
    result = operation.result(timeout=10000)
    print("\nStarting Alignment via GoogleASR for: {}\n".format(audio_file_name))
    for result in result.results:
        alternative = result.alternatives[0]
        transcript_no_offset += alternative.transcript
        print(u'Transcript: {}'.format(alternative.transcript))
        print('Confidence: {}'.format(alternative.confidence))
        interval = 1
        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time
            (start_milliseconds,end_milliseconds) = (start_time.seconds + start_time.nanos * 1e-9, end_time.seconds + end_time.nanos * 1e-9)
            transcribed = ('{}|{}|{}'.format(
                word,
                start_time.seconds + start_time.nanos * 1e-9,
                end_time.seconds + end_time.nanos * 1e-9)) + '\n'
            transcript += transcribed
            alignment += update_textgrid(start_milliseconds,end_milliseconds, word, interval)
            interval += 1
    create_textgrid(textgrid_path, alignment, end_milliseconds, interval-1)
    delete_blob(bucket_name, destination_blob_name)
    print('It took {} seconds.'.format(time.time()-start))
    return (transcript, transcript_no_offset)


def create_textgrid(textgrid_path, alignment, total_time, intervals):
    textgrid = open(textgrid_path,"w+")
    # Write preamble
    textgrid.write("File type = \"ooTextFile\"" + '\n')
    textgrid.write("Object class = \"TextGrid\""+ '\n')
    textgrid.write('\n')
    textgrid.write("xmin = 0.000"+ '\n')
    textgrid.write("xmax = " + str(total_time) +"00" + '\n') #find total time
    textgrid.write("tiers? <exists>"+ '\n')
    textgrid.write("size = 1"+ '\n')
    textgrid.write("item []: "+ '\n')
    textgrid.write("	item [1]:"+ '\n')
    textgrid.write("		class = \"IntervalTier\""+ '\n')
    textgrid.write("		name = \"words\""+ '\n')
    textgrid.write("		xmin = 0.000"+ '\n')
    textgrid.write("		xmax = " + str(total_time) +"00" + '\n') #total time
    textgrid.write("		intervals: size = " + str(intervals) +'\n') # total intervals
    textgrid.write(alignment)
    textgrid.close()


def update_textgrid(start_milliseconds,end_milliseconds, text, interval):
    format = ""
    format += "			intervals [" + str(interval) + "]:" + "\n" #interval number
    format += "				xmin = " + str(start_milliseconds) + "00" + "\n" #current start time
    format += "				xmax = " + str(end_milliseconds) + "00" + "\n" #current end time
    format += "				text = \"" + text + "\" \n" #word
    return format


def write_transcripts(transcript_filename,transcript, OUTPUT_PATH):
    f= open(OUTPUT_PATH + "/" + transcript_filename,"w+")
    f.write(transcript)
    f.close()

def tocsv(transcript_filename, csv_filename, OUTPUT_PATH):
    df=pd.read_csv(OUTPUT_PATH + "/" + transcript_filename,delimiter="|", names=["Word", "Start", "End"])
    df.to_csv(OUTPUT_PATH + "/" + csv_filename, index=False)

for file_name in os.listdir(DIR_PATH):
    if (file_name[-5:] == ".flac"):
        audio_file_name = file_name
        audio_file_nosuffix = audio_file_name.split('.')[0]
        OUTPUT_PATH = DIR_PATH + "/" + audio_file_nosuffix + "_transcripts"
        transcript_filename = audio_file_nosuffix + '.txt'
        full_transcript_filename = audio_file_nosuffix + "_full" + '.txt'
        textgrid_path = OUTPUT_PATH + '/' + audio_file_nosuffix + '.textgrid'
        csv_filename = audio_file_nosuffix + '.csv'
        if not(os.path.isdir(OUTPUT_PATH)):
            print("Starting GoogleASRAligner for {}".format(file_name))
            os.makedirs(OUTPUT_PATH)
            (transcript, transcript_no_offset) = transcribe_file_with_word_time_offsets(audio_file_name, BUCKET_NAME)
            write_transcripts(full_transcript_filename,transcript_no_offset, OUTPUT_PATH)
            write_transcripts(transcript_filename,transcript, OUTPUT_PATH)
            tocsv(transcript_filename, csv_filename, OUTPUT_PATH)