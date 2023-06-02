import { test, expect } from '@playwright/test';

test.describe('User Creation', () => {
  let page;

  test.beforeEach(async ({ browser }) => {
    page = await browser.newPage();
    await page.goto('/users');
    await expect(page.getByText('Users', { exact: true })).toBeVisible();
  });

  test('should find user(s)', async ({ page }) => {
    await page.goto('/users');
    if ((await page.getByText('Loading...')) !== null) {
      await page.getByTestId('user-*');
    }
  });

  test('should create user', async ({ page }) => {
    await page.goto('/users');
    await expect(page.getByTestId('create-user')).toBeVisible();
    // page.on('dialog', dialog => dialog.accept());
    await page.getByTestId('create-user').click();
    // await expect(page.getByText('Create A User Account')).toBeVisible();
  });

});