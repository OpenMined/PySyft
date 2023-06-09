import { error } from '@sveltejs/kit';

/** @type {import('./$types').RequestHandler} */
export function DELETE({ params }) {
  const userId = params.slug;
}
