import { existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tool } from "langchain";
import { createDeepAgent, FilesystemBackend, type SubAgent } from "deepagents";
import { OpenAI } from "openai";
import { z } from "zod";
import 'dotenv/config';

// Configuration -------------------------------------------------------------

const conferenceTheme = "Developer conference named OSS AI Summit: Building with LangChain.";
const numIdeas = 3;
const numStickersPerIdea = 4;
const stickerDir = "stickers";
const ideasFile = "ideas.md";

// Prompts -------------------------------------------------------------------

const stickerSupervisorAgentPrompt = `## Instructions
You manage a creative agency that designs developer conference stickers.

1. First, check if there is an existing list of sticker ideas in the file \`${ideasFile}\` and read it.
    - If not, use the "sticker-ideas-agent" to generate a list of fun sticker ideas by passing on the conference theme. Save the ideas to \`${ideasFile}\`.

2. Once you have the list of sticker ideas, for each idea, use the "sticker-artist-agent" for each idea to design the stickers.

3. After all stickers are created, generate a basic \`index.html\` HTML page that displays a preview of all the stickers as a grid with their ideas as captions. The preview can be viewed by running the command \`npx http-server stickers\`.
`;

const stickerIdeasAgentPrompt = `## Instructions
You are a creative sticker designer for developer conferences. Your goal is to generate unique and engaging sticker ideas based on the given theme.

You must generate a list of ${numIdeas} distinct fun sticker ideas, that are playful, memeable with a tech/geek vibe.
Each idea should be a list item with a basename in kebab-casee, the optional sticker text (or none) and a short description of the visual elements.

## Example format
1. **Code Ninja**
  - Basename: code-ninja
  - Text: "Code Ninja"
  - Visuals: A cartoon ninja character holding a laptop, surrounded by lines of code and binary
2. **Bug Squasher**
   - Basename: bug-squasher
   - Text: none
   - Visuals: A cute bug character being squashed by a giant boot with code snippets flying around
`;

const stickerArtistAgentPrompt = `## Instructions
Create a prompt for gpt-image-1 to generate a print-ready, die-cut developer conference sticker, using the provided sticker idea.

Sticker requirements (must follow EXACTLY):
- Style: bold, clean vector illustration (no photo), tech/developer vibe, energetic, minimal text (avoid long phrases).
- Shape: organic die-cut silhouette around the main graphic (not a square); include an even white border (stroke) approx 6–8% of the sticker's max dimension around the outer contour.
- Background handling (CRITICAL):
  - Transparent background **ONLY outside the die-cut silhouette**.
  - **Inside the silhouette:** a **fully opaque, solid white interior fill (#FFFFFF)** that covers 100% of the sticker shape beneath all artwork.
  - The interior white fill must be continuous and uninterrupted—**no internal transparent pixels, no holes, no see-through areas**, no semi-transparent shading that reveals transparency.
  - Any colors, gradients, or vector elements must sit on top of the solid white interior base.
  - Absolutely no transparent border around the artwork; the white border + white base must fully eliminate internal transparency.
- Colors: vibrant modern conference palette, high contrast, accessible (avoid low-contrast pastel on white).
- Rendering: crisp vector edges, no drop shadows, no photographic textures, no tiny unreadable glyphs.
- Print readiness: design centered, no elements touching the outer edge; border clearly visible; no watermark; no licensing or trademark text.
- Size intent: works when scaled to ~3.4" at 300 DPI (1024px) and still recognizable at 128px.

Explicitly describe in the prompt: transparent background outside the silhouette only; fully opaque solid white interior fill covering the entire sticker shape; plain white die-cut border stroke; vector style; conference developer theme; organic outline; no internal transparency.

After crafting the prompt, use the "create_image" tool with that exact prompt.

Repeat that process ${numStickersPerIdea} times. Sticker are automatically saved to the \`${stickerDir}\` directory.

Finally respond with the number of stickers created and their filenames.

## Tools
You have access to an image creation tool for generating the sticker images.

- \`create_image\`: Use this to generate an image based on a given prompt and basename. Basename must be in kebab-case without numbering, extension or folder path, just the basename for the file.
`;

// Deep Agent Setup ----------------------------------------------------------

console.log('Deep-stickers: configuration: ', { conferenceTheme, numIdeas, numStickersPerIdea, stickerDir, ideasFile });

const createImage = tool(
  async ({ prompt, basename }: { prompt: string; basename: string }) => {
    console.log('create_image: generating image for', basename);
    console.debug('create_image: prompt:', prompt.slice(0, 240));

    const openai = new OpenAI();
    const response = await openai.images.generate({
      prompt,
      model: process.env.IMAGE_MODEL || 'gpt-image-1',
      background: 'opaque',
      size: '1024x1024',
      quality: "auto",
      output_format: "png",
    });

    if (!response.data || response.data.length === 0) {
      console.error('create_image: no image data returned from OpenAI');
      throw new Error("Image generation failed: No image data returned.");
    }

    const imagePath = saveStickerImage(response.data[0].b64_json!, basename);
    console.log(`create_image: image generated and saved -> ${imagePath}`);
    return { imagePath };
  },
  {
    name: "create_image",
    description: "Generate an image based on a given prompt",
    schema: z.object({
      prompt: z.string().describe("Prompt for the image generation"),
      basename: z.string().describe("Basename in snake-case to save the generated image file, without extension, path or numbering"),
    }),
  },
);

const stickerIdeasAgent: SubAgent = {
  name: "sticker-ideas-agent",
  description: "Used to generate sticker ideas based on a conference theme",
  systemPrompt: stickerIdeasAgentPrompt,
};

const stickerArtistAgent: SubAgent = {
  name: "sticker-artist-agent",
  description: "Used to create sticker images based on sticker ideas",
  systemPrompt: stickerArtistAgentPrompt,
  tools: [createImage],
};

const agent = createDeepAgent({
  model: "azure_openai:gpt-5-mini",
  systemPrompt: stickerSupervisorAgentPrompt,
  subagents: [stickerIdeasAgent, stickerArtistAgent],
  backend: (_config) => new FilesystemBackend({
    rootDir: join('.', stickerDir),
    virtualMode: true
  }),
});

try {
  if (!existsSync(stickerDir)) {
    mkdirSync(stickerDir, { recursive: true });
  }

  console.log('Agent: invoking agent...');
  const result = await agent.invoke({
    messages: [{ role: "user", content: `## Conference theme and context\n${conferenceTheme}` }],
  });

  console.log("*** Agent result ***:", result);
  console.log('Agent: completed successfully');
} catch (err) {
  console.error('Agent: invocation failed', err);
  throw err;
}

// Helpers -------------------------------------------------------------------

function getNextStickerNumber(basename: string): number {
  const existingFiles = readdirSync(stickerDir)
    .filter(file => file.startsWith(basename) && file.endsWith(".png"))
    .map(file => {
      const match = file.match(/(?:${basename}-)?(\d+)\.png$/);
      return match ? parseInt(match[1], 10) : 0;
    })
    .sort((a, b) => b - a);
  return existingFiles.length > 0 ? existingFiles[0] + 1 : 1;
}

function saveStickerImage(imageBase64: string, filename: string): string {
  console.log(`saveStickerImage: preparing to save image for filename='${filename}'`);
  const nextNumber = getNextStickerNumber(filename);
  const targetFilename = join(stickerDir, `${filename}-${nextNumber}.png`);

  const imageData = Buffer.from(imageBase64, 'base64');
  writeFileSync(targetFilename, imageData);

  console.log(`saveStickerImage: saved image to ${targetFilename}`);
  return targetFilename;
}
