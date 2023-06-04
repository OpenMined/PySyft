import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';

test.describe('User Sign Up', () => {
  let page: Page;

  test.beforeEach(async ({ browser }) => {
    page = await browser.newPage();
    await page.goto('/signup');
    await expect(page.getByTestId('deployed-on')).toBeVisible();
  });

  test('should display the expected form fields for the sign up page', async () => {
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

  test('should have some fields marked as required', async () => {
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

  test('should successfully register a user', async () => {
    // NOTE: Until we implement Delete user so that we can clean up created test user accounts
    // this will ensure a new user is created and the test will pass
    const testUser = `test-user-${Math.round(Math.random() * 1000)}@gmail.com`;

    const fields = [
      { testid: 'full_name', value: 'Jane Doe' },
      { testid: 'institution', value: 'OpenMined University' },
      { testid: 'email', value: testUser },
      { testid: 'password', value: 'changethis' },
      { testid: 'confirm_password', value: 'changethis' },
      { testid: 'website', value: 'https://openmined.org' }
    ];

    for (const field of fields) {
      await page.getByTestId(field.testid).fill(field.value);
    }

    await page.getByRole('button', { name: /sign up/i }).click();

    await page.waitForURL('**/login');
    await expect(page.getByTestId('deployed-on')).toBeVisible();
    await page.getByTestId('email').fill(testUser);
    await page.getByTestId('password').fill('changethis');
    await page.getByRole('button', { name: /login/i }).click();
    await page.waitForURL('**/datasets');
  });
});
