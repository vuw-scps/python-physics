window.MathJax = {
  tex2jax: {
    inlineMath: [ ["$","$"] ],
    displayMath: [ ["$$","$$"] ]
  }
};

$('code').not('.python').not('.LaTeX').not('.plaintext').addClass('nohighlight');
$('.LaTeX').addClass('plaintext');

hljs.initHighlightingOnLoad();
