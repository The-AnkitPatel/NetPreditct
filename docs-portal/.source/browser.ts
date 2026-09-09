// @ts-nocheck
import { browser } from 'fumadocs-mdx/runtime/browser';
import type * as Config from '../source.config';

const create = browser<typeof Config, import("fumadocs-mdx/runtime/types").InternalTypeConfig & {
  DocData: {
  }
}>();
const browserCollections = {
  docs: create.doc("docs", {"anomaly-vs-prediction.mdx": () => import("../content/docs/anomaly-vs-prediction.mdx?collection=docs"), "api-reference.mdx": () => import("../content/docs/api-reference.mdx?collection=docs"), "architecture.mdx": () => import("../content/docs/architecture.mdx?collection=docs"), "explainability.mdx": () => import("../content/docs/explainability.mdx?collection=docs"), "index.mdx": () => import("../content/docs/index.mdx?collection=docs"), "student-build-journey.mdx": () => import("../content/docs/student-build-journey.mdx?collection=docs"), "temporal-ml.mdx": () => import("../content/docs/temporal-ml.mdx?collection=docs"), "what-if-simulation.mdx": () => import("../content/docs/what-if-simulation.mdx?collection=docs"), }),
};
export default browserCollections;