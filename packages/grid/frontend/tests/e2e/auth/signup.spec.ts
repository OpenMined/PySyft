import { test, expect } from '@playwright/test';

test.describe('User Sign Up', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/signup');
    await expect(page.getByTestId('deployed-on')).toBeVisible();
  });

  test('should display the expected form fields for the sign up page', async ({ page }) => {
    const fieldsTestId = [
      'email',
      'password',
      'full_name',
      'institution',
      'website',
      'confirm_password'
    ];
    for (const field of fieldsTestId) {
      await expect(page.getByTestId(field)).toBeVisible();
    }
  });

  test('should have some fields marked as required', async ({ page }) => {
    const requiredFields = [
      { id: 'fullName', label: 'Full name' },
      { id: 'email', label: 'Email' },
      { id: 'password', label: 'Password' },
      { id: 'confirm_password', label: 'Confirm Password' }
    ];

    for (const field of requiredFields) {
      await expect(page.locator(`input#${field.id}[required]`)).toBeVisible();

      const labelElement = page.locator(`label[for="${field.id}"]`);
      const labelText = await labelElement.textContent();
      const regexString = `${field.label} *`;
      const labelPattern = new RegExp(regexString);

      expect(labelText).toMatch(labelPattern);
    }
  });

  test('should successfully register a user', async ({ page, context }) => {
    // NOTE: Until we implement Delete user so that we can clean up created test user accounts
    // this will ensure a new user is created and the test will pass
    const testUserEmail = `test-user-${Math.round(Math.random() * 1000)}@gmail.com`;
    const testUserPassword = (Math.random() + 1).toString(36).substring(7);

    const testUserCreds = [
      { testid: 'full_name', value: 'Jane Doe' },
      { testid: 'institution', value: 'OpenMined University' },
      { testid: 'email', value: testUserEmail },
      { testid: 'password', value: testUserPassword },
      { testid: 'confirm_password', value: testUserPassword },
      { testid: 'website', value: 'https://openmined.org' }
    ];

    for (const field of testUserCreds) {
      await page.getByTestId(field.testid).fill(field.value);
    }

    await page.getByRole('button', { name: /sign up/i }).click();
    await page.waitForURL('**/login');
    await expect(page.getByTestId('deployed-on')).toBeVisible();

    await page.getByTestId('email').fill(testUserEmail);
    await page.getByTestId('password').fill(testUserPassword);
    await page.getByRole('button', { name: /login/i }).click();

    // intercept storageState
    // get user_id from localStorage

    await page.waitForURL('**/datasets');

    const storageState = await context.storageState();

    console.log(`storageState: ${JSON.stringify(storageState, null, 1)}`);

    // call deleteUser(user_id)
  });

  test('should fail to create user and notify if account/email already exists', async ({
    page
  }) => {
    const existingEmail = 'info@openmined.org';
    const expectedErrMsg = `User already exists with email: ${existingEmail}`;

    const testUserCreds = [
      { testid: 'full_name', value: 'Jane Doe' },
      { testid: 'institution', value: 'OpenMined University' },
      { testid: 'email', value: existingEmail },
      { testid: 'password', value: 'changethis' },
      { testid: 'confirm_password', value: 'changethis' },
      { testid: 'website', value: 'https://openmined.org' }
    ];

    for (const field of testUserCreds) {
      await page.getByTestId(field.testid).fill(field.value);
    }

    await page.getByRole('button', { name: /sign up/i }).click();

    await expect(page.getByTestId('signup_error')).toBeVisible();
    await expect(page.getByTestId('signup_error')).toContainText(expectedErrMsg);
  });
});
