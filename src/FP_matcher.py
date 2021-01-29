# --------------------------------------------------------------------------
# Tensorflow Implementation of Synthetic Fingerprint Generation
# Licensed under The MIT License [see LICENSE for details]
# Written by Young-Mook Kang
# Email: kym343@naver.com
# --------------------------------------------------------------------------

#
# python -m pip install pythonnet==2.3.0
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import clr
import time
import glob
import os
import gc
import math
import numpy as np
from System import IO, Drawing
from System import String, Char, Int32
from System.Collections import *

String.Overloads[Char, Int32]('A', 10)
String.Overloads[Char, Int32]('A', 10)
String.Overloads[Char, Int32]('A', 10)

p = os.getcwd()
print('current path: ', p)
print(clr.__version__)

start_time = time.time()
clr.AddReference(p + '/VeriFinger_SDK/dotNET/Neurotec.dll')
clr.AddReference(p + '/VeriFinger_SDK/dotNET/Neurotec.Biometrics.dll')
clr.AddReference(p + '/VeriFinger_SDK/dotNET/Neurotec.Biometrics.Client.dll')
clr.AddReference(p + '/VeriFinger_SDK/dotNET/Neurotec.Licensing.dll')
clr.AddReference(p + '/VeriFinger_SDK/dotNET/Neurotec.Media.dll')
from Neurotec.Biometrics import (NFinger, NSubject, NTemplateSize, NBiometricStatus, NMatchingSpeed, NBiometricEngine, NTemplate, NFMinutiaFormat)
from Neurotec.Biometrics.Client import NBiometricClient
from Neurotec.Licensing import NLicense
from Neurotec.Images import NImage

def ObtainLicenses():
    print("NLicense:{}".format(NLicense.ObtainComponents("/local", 5000, "Biometrics.FingerExtraction,Biometrics.FingerMatching")))#"/local", 5000, "Biometrics.FingerExtraction,Biometrics.FingerMatching"
    
    if NLicense.ObtainComponents("/local", 5000, "Biometrics.FingerExtraction,Biometrics.FingerMatching") == False:
        return -1
    else:
        return 1

def SingleExtractFromImage(byteImage):
    biometricClient = NBiometricClient()
    subject = NSubject()
    finger = NFinger()
    image = NImage.FromMemory(byteImage)
    # set image properties
    image.HorzResolution = 500
    image.VertResolution = 500
    image.ResolutionIsAspectRatio = False
    # add to subject
    finger.Image = image
    subject.Fingers.Add(finger)
    # extract feature
    biometricClient.FingersTemplateSize = NTemplateSize.Large
    if biometricClient.CreateTemplate(subject) != NBiometricStatus.Ok:
        del biometricClient
        del subject
        del finger
        del image
        gc.collect()
        return -1

    del biometricClient
    del finger
    del image
    gc.collect()
    img_width, img_height, img_quality, minutia_set, _ = NFRecord_to_array(subject)
    return img_width, img_height, img_quality, minutia_set

def SingleExtractFromImage_for_matching(byteImage):
    biometricClient = NBiometricClient()
    subject = NSubject()
    finger = NFinger()
    image = NImage.FromMemory(byteImage)
    # set image properties
    image.HorzResolution = 500
    image.VertResolution = 500
    image.ResolutionIsAspectRatio = False
    # add to subject
    finger.Image = image
    subject.Fingers.Add(finger)
    # extract feature
    biometricClient.FingersTemplateSize = NTemplateSize.Large
    if biometricClient.CreateTemplate(subject) != NBiometricStatus.Ok:
        del biometricClient
        del subject
        del finger
        del image
        gc.collect()
        return -1

    del biometricClient
    del finger
    del image
    gc.collect()
    return subject

def SingleExtractFromFile(filename, save_flag, out_fn):
    biometricClient = NBiometricClient()
    subject = NSubject()
    finger = NFinger()
    image = NImage.FromFile(filename)
    # set image properties
    image.HorzResolution = 500
    image.VertResolution = 500
    image.ResolutionIsAspectRatio = False
    # add to subject
    finger.Image = image
    subject.Fingers.Add(finger)
    # extract feature
    biometricClient.FingersTemplateSize = NTemplateSize.Small
    if biometricClient.CreateTemplate(subject)!= NBiometricStatus.Ok:
        del biometricClient
        del subject
        del finger
        del image
        return -1

    if save_flag == True:
        write_minutiae_to_file(subject, out_fn)

    del biometricClient
    del finger
    del image
    gc.collect()

    img_width, img_height, img_quality, minutia_set, _ = NFRecord_to_array(subject)
    return img_width, img_height, img_quality, minutia_set

def SingleExtractFromFile(filename, save_flag, out_fn):
    biometricClient = NBiometricClient()
    subject = NSubject()
    finger = NFinger()
    image = NImage.FromFile(filename)
    # set image properties
    image.HorzResolution = 500
    image.VertResolution = 500
    image.ResolutionIsAspectRatio = False
    # add to subject
    finger.Image = image
    subject.Fingers.Add(finger)
    # extract feature
    biometricClient.FingersTemplateSize = NTemplateSize.Small
    if biometricClient.CreateTemplate(subject)!= NBiometricStatus.Ok:
        del biometricClient
        del subject
        del finger
        del image
        return -1

    if save_flag == True:
        write_minutiae_to_file(subject, out_fn)

    del biometricClient
    del finger
    del image
    gc.collect()

    img_width, img_height, img_quality, minutia_set, _ = NFRecord_to_array(subject)
    return img_width, img_height, img_quality, minutia_set

def SingleExtractFromFile_for_matching(filename):
    biometricClient = NBiometricClient()
    subject = NSubject()
    finger = NFinger()
    image = NImage.FromFile(filename)
    # set image properties
    image.HorzResolution = 500
    image.VertResolution = 500
    image.ResolutionIsAspectRatio = False
    # add to subject
    finger.Image = image
    subject.Fingers.Add(finger)
    # extract feature
    biometricClient.FingersTemplateSize = NTemplateSize.Small
    if biometricClient.CreateTemplate(subject)!= NBiometricStatus.Ok:
        del biometricClient
        del subject
        del finger
        del image
        return -1

    del biometricClient
    del finger
    del image
    gc.collect()
    return subject

def BatchExtract(directory):
    input_paths = glob.glob(os.path.join(directory, "*.bmp"))
    for file in input_paths:
        print(file)
        if SingleExtractFromFile(file) == -1:
            with open(directory + "\\fail_to_extract.txt", "a") as fid:
                fid.writelines(file)
        gc.collect()

def SingleMatch(subject1, subject2):
    matchingScore = 0
    biometricClient = NBiometricClient()
    biometricClient.MatchingThreshold = 48
    biometricClient.FingersMatchingSpeed = NMatchingSpeed.High
    biometricClient.MatchingWithDetails = True
    status = biometricClient.Verify(subject1, subject2)
    if status == NBiometricStatus.Ok or status == NBiometricStatus.MatchNotFound:
        temp = subject1.MatchingResults.ToArray()
        matchingScore = temp[0].Score

    del biometricClient
    gc.collect()
    return matchingScore

def Match_dFP_realFP(real_fp):
    target = SingleExtractFromFile_for_matching(os.path.join('generated_fp', 'd_fp_tmp-outputs.png'))
    output = SingleExtractFromFile_for_matching(real_fp)
    if target == -1 or output == -1:
        score = 0
    else:
        score = SingleMatch(target, output)
    return score

def single_match_from_file(fn1, fn2):
    flag = True

    target = SingleExtractFromFile_for_matching(fn1)
    output = SingleExtractFromFile_for_matching(fn2)
    if target == -1 or output == -1:
        score = 0
        img_quality_1 = 0
        img_quality_2 = 0

        if output != -1:
            _, _, img_quality_2, _, n_minu_2 = NFRecord_to_array(output)

        if target != -1:
            _, _, img_quality_1, _, n_minu_1 = NFRecord_to_array(target)

        flag = False

    if flag:
        _, _, img_quality_1, _, n_minu_1 = NFRecord_to_array(target)
        _, _, img_quality_2, _, n_minu_2 = NFRecord_to_array(output)
        # print(n_minu_1)
        # print(n_minu_2)
        score = SingleMatch(target, output)

    return score, img_quality_1, img_quality_2

def single_match_from_image(image1, image2):
    target = SingleExtractFromImage_for_matching(image1)
    output = SingleExtractFromImage_for_matching(image2)
    if target == -1 or output == -1:
        score = 0
    else:
        score = SingleMatch(target, output)
    return score

def Matcher_Loss():
    target = SingleExtractFromFile_for_matching("target.png")
    output = SingleExtractFromFile_for_matching("output.png")
    if target == -1 or output == -1:
        score = 0
    else:
        score = SingleMatch(target, output)
    score = float(score)
    if score > 96:
        score = 96
    # normalize score to [0, 1]
    score_loss = math.exp((score-96.0)/96.0)
    return score_loss

def RotationToDegrees(rotation):
    return (2.0 * rotation * 360.0 + 256.0) / (2.0 * 256.0)

def write_minutiae_to_file(subject, out_fn):
    template_buffer = subject.GetTemplateBuffer().ToArray()
    template = NTemplate(template_buffer)

    stream = IO.StreamWriter(out_fn)

    for nfRec in template.Fingers.Records:
        stream.WriteLine("width:\t{0}", nfRec.Width)
        stream.WriteLine("height:\t{0}", nfRec.Height)
        stream.WriteLine("resolution:\t{0}", nfRec.HorzResolution)
        stream.WriteLine("quality:\t{0}", nfRec.Quality)
        stream.WriteLine("nMinutia:\t{0}", nfRec.Minutiae.Count)

        for minutia in nfRec.Minutiae:
            stream.Write("{0}\t", minutia.X)
            stream.Write("{0}\t", minutia.Y)
            stream.Write("{0}\t", RotationToDegrees(minutia.RawAngle))
            stream.Write("{0}\t", minutia.Quality)
            # 1 - ending, 2 - bifurcation
            stream.WriteLine("{0}", minutia.Type)
    stream.Close()

def NFRecord_to_array(subject):
    template_buffer = subject.GetTemplateBuffer().ToArray()
    template = NTemplate(template_buffer)

    for nfRec in template.Fingers.Records:
        img_width = nfRec.Width
        img_height = nfRec.Height
        img_quality = nfRec.Quality
        n_minutiae = nfRec.Minutiae.Count
        minutiaFormat = nfRec.MinutiaFormat

        index = 0
        for minutia in nfRec.Minutiae:
            x = minutia.X
            y = minutia.Y
            direction = RotationToDegrees(minutia.RawAngle)
            if ((minutiaFormat & NFMinutiaFormat.HasQuality) == NFMinutiaFormat.HasQuality):
                quality = minutia.Quality
            else:
                quality = -1
            # 1 - ending, 2 - bifurcation
            type = minutia.Type * 128
            if type > 255:
                type = type -1
            if index == 0:
                minutia_set = [[x, y, direction, quality, type]]
            else:
                minutia_set.append([x, y, direction, quality, type])
            index = index + 1

    return img_width, img_height, img_quality, minutia_set, n_minutiae

