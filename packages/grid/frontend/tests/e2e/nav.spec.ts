import { test, expect } from "@playwright/test"

test("should display syft version in header", async ({ page }) => {
  await page.goto("/")

  // is the version showing?
  const versionText = await page.getByTestId("auth-nav-version").textContent()
  expect(versionText).toBeTruthy()
  expect(versionText).toMatch(/version/i)
})
