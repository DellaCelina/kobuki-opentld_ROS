/*  Copyright 2011 AIT Austrian Institute of Technology
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/
/*
 * MainX.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: Georg Nebehay
 */

#include <sstream>
#include <math.h>
#include "Main.h"

#include "Config.h"
#include "ImAcq.h"
#include "Gui.h"
#include "TLDUtil.h"
#include "Trajectory.h"

#include "ros/ros.h"
#include "geometry_msgs/Twist.h"

using namespace tld;
using namespace cv;

void Main::doWork()
{
	Trajectory trajectory;
    IplImage *img = imAcqGetImg(imAcq);
    Mat grey(img->height, img->width, CV_8UC1);
    cvtColor(cvarrToMat(img), grey, CV_BGR2GRAY);

    CvPoint *pre_point = NULL, *current_point = NULL; 

    tld->detectorCascade->imgWidth = grey.cols;
    tld->detectorCascade->imgHeight = grey.rows;
    tld->detectorCascade->imgWidthStep = grey.step;

	if(showTrajectory)
	{
		trajectory.init(trajectoryLength);
	}

    if(selectManually)
    {

        CvRect box;

        if(getBBFromUser(img, box, gui) == PROGRAM_EXIT)
        {
            return;
        }

        if(initialBB == NULL)
        {
            initialBB = new int[4];
        }

        initialBB[0] = box.x;
        initialBB[1] = box.y;
        initialBB[2] = box.width;
        initialBB[3] = box.height;
    }

    FILE *resultsFile = NULL;

    if(printResults != NULL)
    {
        resultsFile = fopen(printResults, "w");
        if(!resultsFile)
        {
            fprintf(stderr, "Error: Unable to create results-file \"%s\"\n", printResults);
            exit(-1);
        }
    }

    bool reuseFrameOnce = false;
    bool skipProcessingOnce = false;

    if(loadModel && modelPath != NULL)
    {
        tld->readFromFile(modelPath);
        reuseFrameOnce = true;
    }
    else if(initialBB != NULL)
    {
        Rect bb = tldArrayToRect(initialBB);

        printf("Starting at %d %d %d %d\n", bb.x, bb.y, bb.width, bb.height);

        tld->selectObject(grey, &bb);
        skipProcessingOnce = true;
        reuseFrameOnce = true;
    }

	ros::NodeHandle nh;
	ros::Publisher pub = nh.advertise<geometry_msgs::Twist>("mobile_base/commands/velocity", 1);
	geometry_msgs::TwistPtr msg(new geometry_msgs::Twist());
    while(ros::ok() && imAcqHasMoreFrames(imAcq))
    {
        double tic = cvGetTickCount();

        if(!reuseFrameOnce)
        {
            cvReleaseImage(&img);
            img = imAcqGetImg(imAcq);

            if(img == NULL)
            {
                printf("current image is NULL, assuming end of input.\n");
                break;
            }

            cvtColor(cvarrToMat(img), grey, CV_BGR2GRAY);
        }

        if(!skipProcessingOnce)
        {
            tld->processImage(cvarrToMat(img));
        }
        else
        {
            skipProcessingOnce = false;
        }

        if(printResults != NULL)
        {
            if(tld->currBB != NULL)
            {
                fprintf(resultsFile, "%d %.2d %.2d %.2d %.2d %f\n", imAcq->currentFrame - 1, tld->currBB->x, tld->currBB->y, tld->currBB->width, tld->currBB->height, tld->currConf);
            }
            else
            {
                fprintf(resultsFile, "%d NaN NaN NaN NaN NaN\n", imAcq->currentFrame - 1);
            }
        }

        double toc = (cvGetTickCount() - tic) / cvGetTickFrequency();

        toc = toc / 1000000;

        float fps = 1 / toc;

        int confident = (tld->currConf >= threshold) ? 1 : 0;

        if(showOutput || saveDir != NULL)
        {
            char string[128];

            char learningString[10] = "";

            if(tld->learning)
            {
                strcpy(learningString, "Learning");
            }

            sprintf(string, "#%d,Posterior %.2f; fps: %.2f, #numwindows:%d, %s", imAcq->currentFrame - 1, tld->currConf, fps, tld->detectorCascade->numWindows, learningString);
            CvScalar yellow = CV_RGB(255, 255, 0);
            CvScalar blue = CV_RGB(0, 0, 255);
            CvScalar black = CV_RGB(0, 0, 0);
            CvScalar white = CV_RGB(255, 255, 255);

            if(tld->currBB != NULL)
            {
                //CvScalar rectangleColor = (confident) ? blue : yellow;
                //cvRectangle(img, tld->currBB->tl(), tld->currBB->br(), rectangleColor, 8, 8, 0);
                if(pre_point == NULL){
                    pre_point = new CvPoint((tld->currBB->tl().x + tld->currBB->br().x)/2, (tld->currBB->tl().y + tld->currBB->br().y)/2);
                    current_point = new CvPoint(*pre_point);
                }  
                else{
                    current_point = new CvPoint((tld->currBB->tl().x + tld->currBB->br().x)/2, (tld->currBB->tl().y + tld->currBB->br().y)/2);
                }

                cvCircle(img, *pre_point, 3, blue, 3);
                cvCircle(img, *current_point, 3, black, 3);
                cvLine(img, *pre_point, *current_point, white, 2);

                /*float vec, angle;
                vec = sqrt(pow(pre_point->x - current_point->x, 2) + pow(pre_point->y - current_point->y, 2));
                angle = acos((current_point->x - pre_point->x)/vec) * 180/M_PI;

                CvFont font;
				std::stringstream st;
				st << vec << "Force , " << angle << "Degree";
                cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, 1, 1, 0, 2);
                cvPutText(img, st.str().c_str(), CvPoint(50, 150), &font, white); 
                
				*/
				float vel, ang;
				const float vel_step = 200, ang_step = 300;
				vel = pre_point->y - current_point->y;
				vel /= vel_step;
				ang = current_point->x - pre_point->x;
				ang /= ang_step;
				msg->linear.x = vel;
				msg->angular.z = ang;
            }
			else if(showTrajectory)
			{
				trajectory.addPoint(cvPoint(-1, -1), cvScalar(-1, -1, -1));
            }
            else{
                delete pre_point;
                delete current_point;
                pre_point = NULL;
                current_point = NULL;
				msg->linear.x = 0;
				msg->angular.z = 0;
            }

			pub.publish(msg);

			if(showTrajectory)
			{
				trajectory.drawTrajectory(img);
			}

            CvFont font;
            cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, .5, .5, 0, 1, 8);
            cvRectangle(img, cvPoint(0, 0), cvPoint(img->width, 50), black, CV_FILLED, 8, 0);
            cvPutText(img, string, cvPoint(25, 25), &font, white);

            if(showForeground)
            {

                for(size_t i = 0; i < tld->detectorCascade->detectionResult->fgList->size(); i++)
                {
                    Rect r = tld->detectorCascade->detectionResult->fgList->at(i);
                    cvRectangle(img, r.tl(), r.br(), white, 1);
                }

            }


            if(showOutput)
            {
                gui->showImage(img);
                char key = gui->getKey();

                if(key == 'q') break;

                if(key == 'b')
                {

                    ForegroundDetector *fg = tld->detectorCascade->foregroundDetector;

                    if(fg->bgImg.empty())
                    {
                        fg->bgImg = grey.clone();
                    }
                    else
                    {
                        fg->bgImg.release();
                    }
                }

                if(key == 'c')
                {
                    //clear everything
                    tld->release();
                }

                if(key == 'l')
                {
                    tld->learningEnabled = !tld->learningEnabled;
                    printf("LearningEnabled: %d\n", tld->learningEnabled);
                }

                if(key == 'a')
                {
                    tld->alternating = !tld->alternating;
                    printf("alternating: %d\n", tld->alternating);
                }

                if(key == 'e')
                {
                    tld->writeToFile(modelExportFile);
                }

                if(key == 'i')
                {
                    tld->readFromFile(modelPath);
                }

                if(key == 'r')
                {
                    CvRect box;

                    if(getBBFromUser(img, box, gui) == PROGRAM_EXIT)
                    {
                        break;
                    }

                    Rect r = Rect(box);

                    tld->selectObject(grey, &r);
                }
            }

            if(saveDir != NULL)
            {
                char fileName[256];
                sprintf(fileName, "%s/%.5d.png", saveDir, imAcq->currentFrame - 1);

                cvSaveImage(fileName, img);
            }
        }

        if(reuseFrameOnce)
        {
            reuseFrameOnce = false;
        }
    }

    cvReleaseImage(&img);
    img = NULL;

    if(exportModelAfterRun)
    {
        tld->writeToFile(modelExportFile);
    }

    if(resultsFile)
    {
        fclose(resultsFile);
    }
}
