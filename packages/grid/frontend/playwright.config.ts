import { defineConfig, devices } from "@playwright/test"
import * as dotenv from "dotenv"

dotenv.config()

export default defineConfig({
  testDir: "tests/e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: "dot",
  use: {
    baseURL: process.env.TEST_API_URL || "http://localhost:9081",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
})
