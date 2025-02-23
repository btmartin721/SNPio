function process_inlines(inlines)
  local result = {}

  for _, inline in ipairs(inlines) do
    if inline.t == "Str" then
      table.insert(result, inline.text)

    elseif inline.t == "Code" then
      -- Use \verb|...| to safely handle underscores
      local code = inline.text:gsub("|", "\\|")  -- Escape | if it appears inside inline code
      table.insert(result, "\\verb|" .. code .. "|")

    elseif inline.t == "Emph" then
      table.insert(result, "\\textit{" .. process_inlines(inline.content) .. "}")

    elseif inline.t == "Strong" then
      table.insert(result, "\\textbf{" .. process_inlines(inline.content) .. "}")

    elseif inline.t == "Link" then
      table.insert(result, "\\href{" .. inline.target .. "}{" .. process_inlines(inline.content) .. "}")

    else
      table.insert(result, pandoc.utils.stringify(inline))
    end
  end

  return table.concat(result, " ")
end

function process_admonition(elem, box_type)
  local content = process_inlines(elem.content)

  -- Ensure LaTeX correctly wraps multiline text inside the environment
  return pandoc.RawBlock("latex", "\\begin{" .. box_type .. "}\n" .. content .. "\n\\end{" .. box_type .. "}")
end

function Para(elem)
  local content = elem.content

  if #content > 0 then
    local first_element = content[1]

    -- Check for [!NOTE], [!TIP], or [!WARNING]
    if first_element.t == "Str" then
      if first_element.text == "[!NOTE]" then
        table.remove(content, 1)  -- Remove [!NOTE]
        return process_admonition(pandoc.Para(content), "notebox")

      elseif first_element.text == "[!TIP]" then
        table.remove(content, 1)  -- Remove [!TIP]
        return process_admonition(pandoc.Para(content), "tipbox")

      elseif first_element.text == "[!WARNING]" then
        table.remove(content, 1)  -- Remove [!WARNING]
        return process_admonition(pandoc.Para(content), "warningbox")
      end
    end
  end

  return elem
end
