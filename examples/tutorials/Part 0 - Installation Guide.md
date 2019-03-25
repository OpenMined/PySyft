
# Part 0: Installing PySyft

Before you start following the tutorials, you need to install PySyft on your computer.

Here are the PySyft installation instructions for Linux, MacOs and Windows.

Just scroll down to find your preferred OS:

--------------

Authors:

- Alan Aboudib - Twitter: [@alan_aboudib](https://twitter.com/alan_aboudib)



----------------------------
## 0.1 Linux
--------------------

Installing PySyft on Linux is really straight forward. Here are the steps:

#### 1. Make sure you have python >= 3.6

You can check out the version number by running:

<font color='red'>$</font>  `python --version`

#### 2. Install pytorch 1.0:

Get the installation command here:

https://pytorch.org/get-started/locally/. (use the pip option)

and run it in the terminal.

**<font color='red'>Attention:</font>** You might need to use `sudo` to run the installation command if you are not using Conda.


#### 3. Clone the PySyft repo from github

<font color='red'>$</font> `git clone https://github.com/OpenMined/PySyft.git`

#### 4. Enter the cloned repo

<font color='red'>$</font> `cd PySyft`

#### 5. Install PySyft

<font color='red'>$</font> `sudo python setup.py install`

#### 6. Test your installation:

<font color='red'>$</font> `sudo python setup.py test`

-----------------------
## 0.2 Mac OS
-------------------

#### 1. Install python

In order to install python on MacOs, you first need to install **Homebrew** the famous package manager.

Start by opening a terminal and type the following:

<font color='red'>$</font> `xcode-select --install`

<font color='red'>$</font> `/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`

This will install **Homebrew**. Then, install python (version 3.6 or above) by running:

<font color='red'>$</font> `brew install python3`

#### 2. Install pytorch 1.0

You should first get the installation command here:

https://pytorch.org/get-started/locally/. (use the pip option)

Then run that command in the open terminal.

#### 3. Clone the PySyft repo from Github

Just type

<font color='red'>$</font> `git clone https://github.com/OpenMined/PySyft.git`

In the terminal

#### 4. Install PySyft

First, you need to enter the PySyft folder you cloned earlier by typing:

<font color='red'>$</font> `cd PySyft`

in the terminal. Then, run this to install the package:

<font color='red'>$</font> `python setup.py install`

You can  test your installation by running:

<font color='red'>$</font> `python setup.py test`


-----------------------------------
## 0.3 Windows
------------------------------

#### 1. install python

PySyft requires python version 3.6 or above.

Here is the link to install python https://www.python.org/downloads

#### 2. add ***python*** and ***pip*** to the `PATH` environment variable

First, you need to find the folder path to which `python.exe` and `pip3` were installed.

In my case it is:

>C:\Users\alan\AppData\Local\Programs\Python\Python37

for the former, and:

>C:\Users\alan\AppData\Local\Programs\Python\Python37\Scripts

for the latter. So I will add both paths to the PATH environment variable.

In order to do that, go to **Run** and type `sysdm.cpl` and press **Enter**. The following window should open:



<img src='./images/sysdmcpl.png'>



Click on the `Environment Variables...` button in the bottom-right corner. The following window should appear:


<img src='./images/sysdmcpl2.png'>



Select the **Path** row in the `System varibles` section as in the above screenshot and click on `Edit...`

In the window that opens, click on `New` and add the installation paths for `python.exe` and `pip`. Here is an example:



<img src='./images/sysdmcpl3.png'>


Click on `OK`.

#### 3. Install pytorch 1.0

You should first get the installation command here:

https://pytorch.org/get-started/locally/. (use the pip option)

Then open the command prompt by going to **Run** and typing *cmd*, type or paste the pytorch installation command you got from the above link and press **Enter**.

#### 4. Install Git

Here is the download link for Git on windows: https://git-scm.com/download/win

#### 5. Install Microsoft Build tools

Go to https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017

and click on `Free download` under **Community** in the Visual Studio download section. This is illustrated in the following screenshot:




<img src='./images/visual_sudio_download.png'>



After the download is finished, run the downloaded package. You will eventually get the following window:



<img src='./images/microsoft_build_tools_choice.png'>



Select `Desktop development with C++` and click on `Install` at the bottom-right corner of the page. (In the above screenshot you see a `Close` button instead since I have already installed it.)

Wait until the the install has finished.

#### 6. Clone the PySyft repo from Github

Just type

<font color='red'>**>**</font>`git clone https://github.com/OpenMined/PySyft.git`

In the command prompt.

#### 7. Install PySyft

First, you need to enter the PySyft folder you cloned earlier by typing:

<font color='red'>**>**</font>`cd PySyft`

in the command prompt.

Then, type this in the command prompt as well:

<font color='red'>**>**</font> `python setup.py install`

This will install PySyft to your system. You should see no errors.

You can also test your installation by running:

<font color='red'>**>**</font>`python setup.py test`

--------------
-------------
## 0.4 Troubleshooting
----------------------

If installation is not working for you, please tell us about your issue on our slack channel here:

<a href='https://openmined.slack.com'>openmined.slack.com</a>

We will do our best to help !!
