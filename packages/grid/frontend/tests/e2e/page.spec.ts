import { test, expect } from "@playwright/test"

test("should display the page correctly", async ({ page }) => {
  await page.goto("/")
  const title = await page.title()
  expect(title).toBe("Syft UI")
})

test("should navigate to login page", async ({ page }) => {
  await page.goto("/login")
  const title = await page.title()
  expect(title).toBe("Syft UI")
})
