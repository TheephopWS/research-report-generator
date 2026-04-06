FROM python:3.12-slim

# Install pdflatex and the LaTeX packages used by the report template.
# texlive-latex-extra covers: setspace, parskip, booktabs, geometry, hyperref
# texlive-fonts-recommended + lmodern cover the lmodern font package
RUN apt-get update && apt-get install -y --no-install-recommends \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
        lmodern \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
