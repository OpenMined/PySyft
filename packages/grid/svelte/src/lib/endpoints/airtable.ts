const AIRTABLE_BASE_ID = import.meta.env.VITE_AIRTABLE_BASE_ID;
const AIRTABLE_TOKEN = import.meta.env.VITE_AIRTABLE_TOKEN;
const AIRTABLE_URL = `https://api.airtable.com/v0/${AIRTABLE_BASE_ID}/checklist-data`;

export type PostReqBody = { fields: { item: string } };

export const storeChecklistItems = async (reqBody: PostReqBody) => {
  const res = await fetch(AIRTABLE_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${AIRTABLE_TOKEN}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(reqBody)
  });

  if (res.ok) {
    return {
      status: 200,
      body: {
        message: 'success'
      }
    };
  } else {
    return {
      status: 404,
      body: {
        message: 'failed'
      }
    };
  }
};
