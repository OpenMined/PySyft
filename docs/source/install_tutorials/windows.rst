.. _windows_install:

=================
Windows Tutorials
=================

The following instructions are for Windows 10 version 2004 or higher.

Now, traditionally, getting things as big and imposing as PySyft to work on Windows is... really, really challenging.
Luckily for us, we've got a few tricks up our sleeves to make the process super easy.

So sit back, relax, grab a few cookies, and *enjoy!*

Step 1: Enabling WSL2
=====================

Our first and most important step is going to be to enable the Windows Subsystem for Linux (WSL).
This lets you run a Linux-based environment (including most command line tools and applications!) directly on Windows,
unmodified, and without any of the drawbacks of more traditional solutions like virtual machines or dual-booting.


Installing this incredible piece of software is as easy as opening PowerShell or Command Prompt in the Start Menu, and entering::

    wsl --install

And that's it! It'll start installing all the dependencies and getting things in order.
If you run into any issues here, please refer to `this link <https://docs.microsoft.com/en-us/windows/wsl/troubleshooting#installation-issues>`_, which covers common WSL installation issues.

.. Specifying an alternate way to install wsl along with distro from microsoft store start
**Alternate way**
=================

**Install WSL from Microsoft Store**
If the command line has you feeling confused, fear not! There's a more user-friendly approach to installing WSL on Windows. We can bypass the command line altogether and download a package of all the components from the Microsoft Store. Not only that, but this method runs WSL isolated from Windows 11 and updates will be available through the Microsoft Store, so you won't have to wait for the next version of the operating system to install the newest version.

To install WSL from the Microsoft Store, use these steps:


1. Enable Virtual Machine Platform
==================================

 - Open **Start**
 - Search for **Turn Windows Features on or off** and click the  
   top result to open the app 
 - Check the **Virtual Machine Platform**
 - Click the **OK** button 
 - Click the **Restart button**

After completing these steps, you can download the app from the Microsoft Store.

 2. Install Windows Subsystem for Linux app
 ==========================================

- Open the `Windows Subsystem for Linux Store Page <https://www.microsoft.com/en-us/p/windows-subsystem-for-linux-preview/9p9tqf7mrm4r?activetab=pivot:overviewtab>`_
- Click the **Get** button 
- Click the **Open** button 
- Click the **Get** button again

 3. Install Linux Distro
 =======================
- Open **Microsoft Store** app.
- Search for Linux distro. For example `Ubuntu <https://apps.microsoft.com/store/detail/ubuntu-22041-lts/9PN20MSR04DW>`_`
- Click the **Get** button.
- Click the **Open** button.

*Congratulations! Once you complete the steps, WSL will install on Windows 11, including the support for Linux GUI apps and the Linux distribution.*

*To access the command line for your Linux distribution, search for "wsl" in the search bar and select the top result, which should be a penguin logo*

 .. end

Step 2: Setting up Linux User Info
==================================

Well done! You've *almost* got an entire Linux kernel and distribution on your machine, and you did this with **barely one line of code!**
There's just one last step needed. And luckily for us, it's an easy one...

We now have to add a new User to our brand new and shiny Linux distro. To do this, we'll have to pick a username and password.
Please note- this account, this password- doesn't have any relation with your regular Windows username or password. It's specific to the Linux
distro that you just installed.

Once you provide a username and password, **congratulations!** You have a fully fledged Linux distro. You may not have realized it, but you've just unlocked
a whole new universe of possibilities and interesting tools.

Step 3: Updating & Upgrading
============================

Now that you have a shiny new copy of Linux, your next step will be to update and upgrade it.
This is pretty easy to do in Linux, and it's something we can do with *just one command!*

In your new Ubuntu terminal, enter the following command::

    sudo apt update && sudo apt upgrade

You might need to enter the password of the account you created in Step 2. You might also need to press Y and hit enter to allow the updates.
But you're on a roll- nothing will stop you from getting the most up-to-date, and secure version of your Linux distro!

Note: We'd actually recommend doing this reasonably often (once every few days) to maintain a safe and up-to-date distro.

Optional: Installing Windows Terminal
=====================================

We'd recommend installing the Windows Terminal, and using that to launch your Linux Distribution instead of PowerShell, Command Prompt, or the default
Ubuntu shell that comes bundled in.

This isn't strictly necessary, but it doesn't take too long, improves the command line experience, and will probably make you happier.

Please go `here <https://docs.microsoft.com/en-us/windows/terminal/install>`_ if you're interested.

Step 4: Installing Conda
========================

Wow! We've made it pretty far together in a pretty short amount of time.

We've already installed a Linux distribution, (and if you followed the Optional step, have a *swanky* new terminal!) and we're getting *really* close to installing our software.
Our next step is an important one. It'll help us make sure our software can install without any conflicts, and once installed, that it will be stable, and work as intended!

We're going to use a tool called Anaconda to do this. It'll help us create something called a "Virtual Environment."

To install Anaconda, please follow the yellow brick road I lay down here below:

- `Head to the Anaconda website <https://www.anaconda.com/products/individual#Downloads>`_, and find the latest Linux installer.
- Right click the installer, and select **"Copy Link Address"**
- Head back to your WSL terminal, and type "wget " and then right click next to it. This should paste the link you copied, which should produce something like::

    wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

- You got it! Not only did you get it, you made it look **easy.** Now just hit enter.
- At this point, Conda will start installing. Type "yes" and hit Enter for all the various prompts that follow (Accepting the Terms and Conditions, Running Conda Init, etc)
- Once this is done, close and restart your WSL terminal.
- Once restarted, verify that conda is working using the following command::

    conda env list

Wait wait wait wait just a second.
Do you realize what just happened?

You've just successfully installed Anaconda!! Hooray!
Trust me, your life is about to become a LOT easier.


- Let's now tap into your newfound powers with Anaconda and create a new virtual environment called "syft_env" by running the following in your WSL shell::

    conda create -n syft_env python=3.9 --y

- Let's verify that we created our "syft_env" successfully with the following command (Deja Vu, anyone?)::

    conda env list

- You should see two environments in the output. Hooray! Now let's activate the syft virtual env, and let the fun *really* begin::

    conda activate syft_env

- Now let's use it to conveniently install a few packages::

    sudo apt install python3-pip
    pip3 install pandas matplotlib numpy
    pip3 install jupyterlab

- If the last command fails, try the following instead::

    conda install -c conda-forge jupyterlab


Step 5: Become the Docker Doctor
================================

The last tool needed to complete your arsenal is called Docker.
You can install it by following the instructions `here <https://docs.docker.com/desktop/windows/install/>`_.

Note: The windows user account that launches wsl 2 has to be added to the local group "docker-users". On Windows 10 Home, run netplwiz to add the Windows user to the group "docker-users".

Once you have it running, you just have to ensure the following:
- You've allocated a sufficient amount of RAM (we recommend atleast 8GB, but you can get by with less)
- You're using the WSL2 backend

Congratulations, you have reached the end of your journey. Now it is time for your **ultimate test!** Deploying a domain node.

Note that your ultimate test is **optional**- you can do this part later.


Step 6: Install Hagrid and PySyft
=================================

- With the power of WSL and Anaconda, installing our software is as easy as::

    pip3 install syft
    pip3 install hagrid


Optional: Deploy a Domain Node!
===============================

Everything we've done so far has been to make this next part as easy as possible. This is the moment we've all been waiting for.

To launch a domain node called "test_domain", ensure your Virtual Environment ("syft_env" in the steps above) is active, that Docker Desktop is running, and run the command below on your WSL terminal::

    hagrid launch test_domain

Note: If you get the error message "test_domain is not valid for node_type please use one of the following options: ['domain', 'network']" then rerun the command by changing test_domain to domain.

You should see the containers begin to appear on Docker!

**CONGRATULATIONS!!!**

You have reached the promise land. You're ready to begin remote data science.
It was a pleasure walking you through the installation process. Now be sure to use your newfound powers and abilities for good!
