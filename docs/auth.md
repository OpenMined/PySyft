# Authentication

**If you're using Google Colab, authentication is handled by colab via a browser pop-up when you initialize a SyftClient, you can stop reading here.** Once authenticated, Syft Client uses Google Drive as its communication protocol — all messages, events, and files are synced through the Drive API.

## Local / Jupyter Lab Setup

To use Syft Client outside of Google Colab, you need to set up a Google Cloud project with OAuth credentials.

## Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click **Click a project** in the top navigation bar
3. Click **New Project** in the dialog that appears
4. Enter a project name (e.g., "Syft Client")
5. Click **Create**
6. Wait for the project to be created, then select it

## Step 2: Enable the API's

1. In your project, go to **APIs & Services** > **Library**
2. Search for "Google Drive API"
3. Click on **Google Drive API**
4. Click **Enable**
5. In case you are using notifications (Data owner only), do the same for Gmail
6. In case you are using email approval (Data owner only), do the same for Cloud Pub/Sub API

## Step 3: Configure OAuth Consent Screen

1. Go to **APIs & Services** > **OAuth consent screen**
2. press **Get started**
3. Fill in the required fields:
   - **App name**: "Syft Client" (or your preferred name)
   - **User support email**: Your email address
   - click **next**
4. Select **External** user type (unless you have a Google Workspace organization), click next
5. Fill in your email, click **next**
6. Mark the policy checkbox, click **Continue** and **Create**
7. On the **data access** section for the Oauth Consent screen. As a data scientist, you only ever need gdrive, but if you are using syft-client as a data owner, you may also need Gmail, specifically if you use email features like notifications or email approval.
   - GDrive:
     - Click **Add or Remove Scopes**
     - Search for and select `https://www.googleapis.com/auth/drive`
     - Scroll down and click **Update**
     - Scroll down and click **Save**
   - Gmail (data owner only for email features):
     - Click **Add or Remove Scopes**
     - Search for and select `https://www.googleapis.com/auth/gmail.modify`
     - Scroll down and click **Update**
     - Scroll down and click **Save**
8. On the **Audience** section for the oauth consent screen under **Test users**:
   - Click **Add Users**
   - Add you email adress
   - Click **Save and Continue**

## Step 4: Create OAuth Client Credentials

1. In the main navigation menu, go to **APIs & Services** > **Credentials**
2. In the top bar, click **Create Credentials** > **OAuth client ID**
3. Select **Desktop app** as the application type
4. Enter a name (e.g., "Syft Client Desktop")
5. Click **Create**
6. **Download the JSON file** - this contains your client credentials
7. Save this file securely (e.g., as `credentials.json`)

## Step 5: Publish the App

**If your app is Interal (see step 3.4) you can skip this step. **If your app is external and not published (i.e., remains in "Testing" mode), OAuth tokens expire every 7 days and users will need to re-authenticate. Publishing the app removes this limitation.

1. Go to **APIs & Services** > **OAuth consent screen**
2. navigate to the **Audience** section
3. Under **Testing** header, Click **Publish App**
4. Click **confirm**

> **Note:** Publishing may require verification for apps requesting sensitive scopes like Google Drive access.

## Generating a Token

Once you've completed the Google Cloud Console setup, generate a token and log in:

```bash
token_path = sc.credentials_to_token(credentials_path)
do_client = login_do(email="your@email.com", token_path=token_path) # or login_ds
```
