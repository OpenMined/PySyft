import { test, expect } from '@playwright/test';

test.describe('User Login', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
    await expect(page.getByTestId('deployed-on')).toContainText(/deployed on/i);
  });

  test('should display the expected form fields for the login page', async ({ page }) => {
    const fieldsTestId = ['email', 'password'];
    for (const field of fieldsTestId) {
      await expect(page.getByTestId(field)).toBeVisible();
    }
  });

  test('should have some fields marked as required', async ({ page }) => {
    const requiredFields = [
      { id: 'email', label: 'Email' },
      { id: 'password', label: 'Password' }
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

  test('should login an existing user', async ({ page }) => {
    await page.getByTestId('email').fill('info@openmined.org');
    await page.getByTestId('password').fill('changethis');

    await page.getByRole('button', { name: /login/i }).click();

    await page.waitForURL('**/datasets**');
  });

  test('should fail to login a non-existing user', async ({ page }) => {
    await page.getByTestId('email').fill('unregistered_user@openmined.org');
    await page.getByTestId('password').fill('badpass');

    await page.getByRole('button', { name: /login/i }).click();

    await expect(page.getByTestId('login_error')).toBeVisible();
  });
});
