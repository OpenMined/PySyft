import { test, expect } from '@playwright/test';

test.describe('User Creation', () => {
  let page;

  test.beforeEach(async ({ browser }) => {
    page = await browser.newPage();
    await page.goto('/users');
    await expect(page.getByText('Users', { exact: true })).toBeVisible();
  });

  test('should find users', async ({ page }) => {
    await page.goto('/users');

  });

  test('should create user', async ({ page }) => {
    await page.goto('/users');

  });

});