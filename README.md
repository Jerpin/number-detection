# number-detection
This is a task uses raspberry pi with camera to recognize a number. Raspberrypi is set on a ship which is going to sail to where the number is.

There will be three numbers, which are 1, 13, and 3. Only 1 and 3 were to be recognized. 

'M' on 13 means multiple numbers.

<img width="639" alt="截圖_2022-03-30_下午11_06_02" src="https://user-images.githubusercontent.com/57026482/160869188-e0b9df98-666b-47a7-82c2-6ede8beddbd2.png">


setps:
1. original image
<img width="75" alt="image" src="https://user-images.githubusercontent.com/57026482/160864538-3484d75f-2096-43cb-a0db-bb008bdd0b6a.png">

2. remove background
<img width="53" alt="image" src="https://user-images.githubusercontent.com/57026482/160864606-4dfacbc1-21f8-4616-be5c-b1df8035dd77.png">

3. remove background with algorithm
<img width="62" alt="image" src="https://user-images.githubusercontent.com/57026482/160864967-926ccf5b-b689-4ad2-98ca-5e898041ccd1.png">

4. erode to fit the training data

training data(MNIST)

<img width="77" alt="image" src="https://user-images.githubusercontent.com/57026482/160865129-1b636fef-1ac2-48f3-b18a-f9f03caf63ee.png">

image after erosion

<img width="71" alt="image" src="https://user-images.githubusercontent.com/57026482/160865138-c9655a77-6cd5-440e-a1e5-3ca3f2fbcf51.png">
